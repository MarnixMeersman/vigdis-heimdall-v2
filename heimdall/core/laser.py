"""
LaserCube UDP Controller

A robust Python library for controlling Wicked Lasers LaserCube devices
over UDP network protocol. Supports discovery, connection, and real-time point streaming.

Based on the LaserDock protocol:
https://github.com/Wickedlasers/libLaserdockCore/blob/master/3rdparty/laserdocklib/src/LaserDockNetworkDevice.cpp
"""

import logging
import select
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Dict, Any
import queue


# UDP Ports (from LaserDock protocol)
ALIVE_PORT = 45456      # For alive/ping messages
CMD_PORT = 45457        # For commands (info, enable/disable, settings)
DATA_PORT = 45458       # For actual laser point data

# Protocol Commands (from C++ reference)
class Commands:
    GET_ALIVE = 0x27
    GET_FULL_INFO = 0x77
    ENABLE_BUFFER_SIZE_RESPONSE_ON_DATA = 0x78
    SET_OUTPUT = 0x80
    SET_ILDA_RATE = 0x82
    GET_RINGBUFFER_EMPTY_SAMPLE_COUNT = 0x8a
    CLEAR_RINGBUFFER = 0x8d
    SET_NV_MODEL_INFO = 0x97
    SET_DAC_BUF_THOLD_LVL = 0xa0
    SECURITY_CMD_REQUEST = 0xb0
    SECURITY_CMD_RESPONSE = 0xb1
    SAMPLE_DATA = 0xa9
    SAMPLE_DATA_COMPRESSED = 0x9a

# Output enable/disable values
OUTPUT_ENABLE = 0x01
OUTPUT_DISABLE = 0x00

# Protocol constants
MIN_PROTOCOL_VER = 0
COMMS_TIMEOUT_MS = 4000
INACTIVE_INFO_REQUEST_PERIOD_MS = 250
ACTIVE_INFO_REQUEST_PERIOD_MS = 2500
MAX_UDP_PACKETS = 20
MAX_SAMPLES_PER_UDP_PACKET = 140
COMMAND_REPEAT_COUNT = 2
SECURITY_TIMEOUT_MS = 1000


@dataclass
class LaserInfo:
    """Information about a connected LaserCube device."""
    model_name: str
    fw_major: int
    fw_minor: int
    output_enabled: bool
    interlock_enabled: bool
    temperature_warning: bool
    over_temperature: bool
    packet_errors: int
    dac_rate: int
    max_dac_rate: int
    rx_buffer_free: int
    rx_buffer_size: int
    battery_percent: int
    temperature: int
    connection_type: int
    model_number: int
    serial_number: str
    ip_addr: str

    def __str__(self):
        return (f"LaserCube {self.model_name} v{self.fw_major}.{self.fw_minor} "
                f"@ {self.ip_addr} (Battery: {self.battery_percent}%)")


class LaserPoint:
    """Represents a single laser point with position and color."""
    
    def __init__(self, x: int, y: int, r: int, g: int, b: int):
        # Clamp values to valid range (0-4095) for 12-bit DAC
        self.x = max(0, min(4095, x))
        self.y = max(0, min(4095, y))
        self.r = max(0, min(4095, r))
        self.g = max(0, min(4095, g))
        self.b = max(0, min(4095, b))
    
    def to_bytes(self) -> bytes:
        """Convert point to binary format for transmission."""
        # Use 12-bit values directly (0-4095) like mouse_controlled_laser.py
        return struct.pack('<HHHHH', self.x, self.y, self.r, self.g, self.b)


class LaserCube:
    """
    Represents a single LaserCube device and handles communication with it.
    
    Args:
        addr: IP address of the LaserCube
        frame_generator: Function that returns a list of LaserPoint objects for each frame
        cmd_sock: Command socket for this device
        data_sock: Data socket for this device
    """
    
    def __init__(self, addr: str, frame_generator: Callable[[], List[LaserPoint]], 
                 cmd_sock: socket.socket, data_sock: socket.socket):
        self.addr = addr
        self.frame_generator = frame_generator
        self.info: Optional[LaserInfo] = None
        self.remote_buf_free = 0
        self.running = False
        self.authenticated = False
        self.disconnected = False
        
        # Socket references
        self.cmd_sock = cmd_sock
        self.data_sock = data_sock
        
        # Threading
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Buffer management
        self._sample_queue = queue.Queue(maxsize=8000)
        self._message_num = 0
        self._frame_num = 0
        
        # Timing
        self._last_buffer_update = time.time()
        self._info_request_timer = time.time()
        self._comms_timeout = time.time() + COMMS_TIMEOUT_MS / 1000.0
        
        self._logger = logging.getLogger(f"LaserCube[{addr}]")
        self._logger.info(f"Created LaserCube instance for {addr}")
    
    def start(self) -> None:
        """Start the laser control thread."""
        if not self.running:
            self.running = True
            self.disconnected = False
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._main_loop, daemon=True)
            self._thread.start()
            self._logger.info("Started laser control thread")
    
    def stop(self) -> None:
        """Stop the laser control thread and disable output."""
        if self.running:
            self.running = False
            self._stop_event.set()
            
            # Disable output before stopping
            self._send_command([Commands.SET_OUTPUT, OUTPUT_DISABLE])
            self._send_command([Commands.ENABLE_BUFFER_SIZE_RESPONSE_ON_DATA, 0x00])
            
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=2.0)
            
            self._logger.info("Stopped laser control thread")
    
    def is_connected(self) -> bool:
        """Check if the laser is connected and responsive."""
        return self.info is not None and not self.disconnected
    
    def enable_output(self) -> bool:
        """Enable laser output."""
        if not self.is_connected():
            return False
        
        self._send_command([Commands.SET_OUTPUT, OUTPUT_ENABLE])
        self._send_command([Commands.ENABLE_BUFFER_SIZE_RESPONSE_ON_DATA, 0x01])
        self._logger.info("Laser output enabled")
        return True
    
    def disable_output(self) -> bool:
        """Disable laser output."""
        self._send_command([Commands.SET_OUTPUT, OUTPUT_DISABLE])
        self._send_command([Commands.ENABLE_BUFFER_SIZE_RESPONSE_ON_DATA, 0x00])
        self._logger.info("Laser output disabled")
        return True
    
    def clear_buffer(self) -> bool:
        """Clear the laser's ring buffer."""
        self._send_command([Commands.CLEAR_RINGBUFFER])
        return True
    
    def set_dac_rate(self, rate: int) -> bool:
        """Set the DAC sample rate."""
        if rate <= 0:
            return False
        
        rate_bytes = struct.pack('<I', rate)
        cmd = [Commands.SET_ILDA_RATE] + list(rate_bytes)
        self._send_command(cmd)
        self._logger.info(f"Set DAC rate to {rate}")
        return True
    
    def set_buffer_threshold(self, level: int) -> bool:
        """Set the DAC buffer threshold level."""
        if self.info and level >= self.info.rx_buffer_size:
            return False
        
        # Only supported on model 10+ or firmware > 1.23
        if (self.info and 
            (self.info.model_number >= 10 or 
             self.info.fw_major > 1 or 
             self.info.fw_minor > 23)):
            
            level_bytes = struct.pack('<I', level)
            cmd = [Commands.SET_DAC_BUF_THOLD_LVL] + list(level_bytes)
            self._send_command(cmd)
            self._logger.info(f"Set buffer threshold to {level}")
            return True
        
        return False
    
    def process_message(self, msg: bytes) -> None:
        """Process incoming message from the laser."""
        if not msg:
            return
        
        # Reset communication timeout
        self._comms_timeout = time.time() + COMMS_TIMEOUT_MS / 1000.0
        self.disconnected = False
        
        cmd = msg[0]
        
        if cmd == Commands.GET_FULL_INFO:
            self._parse_info_message(msg)
        elif cmd == Commands.GET_RINGBUFFER_EMPTY_SAMPLE_COUNT:
            self._parse_buffer_status(msg)
        elif cmd == Commands.ENABLE_BUFFER_SIZE_RESPONSE_ON_DATA:
            self._logger.debug("Buffer size response acknowledged")
        elif cmd == Commands.SET_OUTPUT:
            self._logger.debug("Output command acknowledged")
        elif cmd == Commands.SECURITY_CMD_RESPONSE:
            self._parse_security_response(msg)
    
    def _parse_info_message(self, msg: bytes) -> None:
        """Parse the full info message from the laser."""
        if len(msg) < 64:
            self._logger.warning(f"Info message too short: {len(msg)} bytes")
            return
        
        try:
            # Parse protocol version
            payload_ver = msg[2]
            if payload_ver < MIN_PROTOCOL_VER:
                self._logger.warning(f"Unsupported protocol version: {payload_ver}")
                return
            
            # Parse firmware version
            fw_major = msg[3]
            fw_minor = msg[4]
            
            # Parse flags
            flags = msg[5]
            output_enabled = bool(flags & 1)
            interlock_enabled = bool(flags & 2) if fw_major > 0 or fw_minor >= 13 else bool(flags & 8)
            temp_warning = bool(flags & 4) if fw_major > 0 or fw_minor >= 13 else bool(flags & 16)
            over_temp = bool(flags & 8) if fw_major > 0 or fw_minor >= 13 else bool(flags & 32)
            packet_errors = (flags >> 4) & 0x0f if fw_major > 0 or fw_minor >= 13 else 0
            
            # Parse DAC rates
            dac_rate = struct.unpack('<I', msg[10:14])[0]
            max_dac_rate = struct.unpack('<I', msg[14:18])[0]
            
            # Parse buffer info
            buffer_free = struct.unpack('<H', msg[19:21])[0]
            buffer_size = struct.unpack('<H', msg[21:23])[0]
            
            # Parse device info
            battery_percent = msg[23]
            temperature = struct.unpack('b', msg[24:25])[0]  # signed byte
            connection_type = msg[25] + 1
            model_number = msg[37]
            
            # Parse serial number
            serial_bytes = msg[26:32]
            serial_number = ':'.join(f'{b:02x}' for b in serial_bytes)
            
            # Parse IP address
            ip_bytes = msg[32:36]
            ip_addr = '.'.join(str(b) for b in ip_bytes)
            
            # Parse model name
            name_start = 38
            name_end = msg.find(b'\x00', name_start)
            if name_end == -1:
                name_end = len(msg)
            model_name = msg[name_start:name_end].decode('utf-8', errors='ignore')
            
            # Create info object
            info = LaserInfo(
                model_name=model_name,
                fw_major=fw_major,
                fw_minor=fw_minor,
                output_enabled=output_enabled,
                interlock_enabled=interlock_enabled,
                temperature_warning=temp_warning,
                over_temperature=over_temp,
                packet_errors=packet_errors,
                dac_rate=dac_rate,
                max_dac_rate=max_dac_rate,
                rx_buffer_free=buffer_free,
                rx_buffer_size=buffer_size,
                battery_percent=battery_percent,
                temperature=temperature,
                connection_type=connection_type,
                model_number=model_number,
                serial_number=serial_number,
                ip_addr=ip_addr if ip_addr != "0.0.0.0" else self.addr
            )
            
            # Update info if changed
            if info != self.info:
                old_info = self.info
                self.info = info
                self.remote_buf_free = info.rx_buffer_free
                
                if old_info is None:
                    self._logger.info(f"Connected to {info}")
                else:
                    self._logger.debug(f"Updated laser info: {info}")
                
        except (struct.error, IndexError) as e:
            self._logger.error(f"Failed to parse info message: {e}")
    
    def _parse_buffer_status(self, msg: bytes) -> None:
        """Parse buffer status message."""
        if len(msg) < 4:
            self._logger.warning(f"Buffer status message too short: {len(msg)} bytes")
            return
        
        try:
            # Check if this is a successful response
            if msg[1] != 0x00:
                self._logger.warning(f"Buffer status command failed: {msg[1]}")
                return
            
            buffer_free = struct.unpack('<H', msg[2:4])[0]
            self.remote_buf_free = buffer_free
            self._logger.debug(f"Buffer free: {buffer_free}")
            
        except struct.error as e:
            self._logger.error(f"Failed to parse buffer status: {e}")
    
    def _parse_security_response(self, msg: bytes) -> None:
        """Parse security response message."""
        if len(msg) < 3:
            return
        
        success = msg[1] == 0x00 and msg[2] == 0x00
        response_data = msg[3:]
        
        if success:
            self.authenticated = True
            self._logger.info("Authentication successful")
        else:
            self._logger.warning("Authentication failed")
    
    def _send_command(self, cmd: List[int]) -> None:
        """Send a command to the laser (sent multiple times for reliability)."""
        if not self.running:
            return
        
        cmd_data = bytes(cmd)
        try:
            # Send multiple times for reliability
            for _ in range(COMMAND_REPEAT_COUNT):
                self.cmd_sock.sendto(cmd_data, (self.addr, CMD_PORT))
        except OSError as e:
            self._logger.error(f"Failed to send command {cmd}: {e}")
    
    def _request_info(self) -> None:
        """Request full info from the laser."""
        self._send_command([Commands.GET_FULL_INFO])
    
    def _request_buffer_status(self) -> None:
        """Request buffer status from the laser."""
        self._send_command([Commands.GET_RINGBUFFER_EMPTY_SAMPLE_COUNT])
    
    def _check_timeouts(self) -> None:
        """Check for communication timeouts."""
        current_time = time.time()
        
        # Check communication timeout
        if current_time > self._comms_timeout:
            if not self.disconnected:
                self._logger.warning("Communication timeout - device disconnected")
                self.disconnected = True
                # Clear sample queue on disconnect
                try:
                    while True:
                        self._sample_queue.get_nowait()
                except queue.Empty:
                    pass
        
        # Periodic info requests
        info_period = (ACTIVE_INFO_REQUEST_PERIOD_MS if self.info and self.info.output_enabled 
                      else INACTIVE_INFO_REQUEST_PERIOD_MS) / 1000.0
        
        if current_time - self._info_request_timer > info_period:
            self._request_info()
            self._info_request_timer = current_time
    
    def _estimate_buffer_recovery(self) -> None:
        """Estimate buffer recovery based on DAC rate."""
        current_time = time.time()
        time_elapsed = current_time - self._last_buffer_update
        
        if self.info and self.info.dac_rate > 0 and time_elapsed > 0:
            # Estimate samples consumed based on DAC rate
            samples_consumed = int(time_elapsed * self.info.dac_rate)
            if samples_consumed > 0:
                self.remote_buf_free = min(
                    self.remote_buf_free + samples_consumed,
                    self.info.rx_buffer_size
                )
        
        self._last_buffer_update = current_time
    
    def _main_loop(self) -> None:
        """Main laser control loop."""
        self._logger.info("Starting main control loop")
        
        # Initial setup
        self._request_info()
        
        try:
            while self.running and not self._stop_event.is_set():
                # Check timeouts and connection status
                self._check_timeouts()
                
                if self.disconnected:
                    time.sleep(0.1)
                    continue
                
                # Update buffer estimate
                self._estimate_buffer_recovery()
                
                # Get next frame from generator
                try:
                    current_frame = self.frame_generator()
                except Exception as e:
                    self._logger.error(f"Frame generator error: {e}")
                    time.sleep(0.01)
                    continue
                
                if not current_frame:
                    time.sleep(0.01)
                    continue
                
                # Send frame data
                self._send_frame_data(current_frame)
                
                # Small delay to prevent overwhelming the laser
                time.sleep(0.001)
                
        except Exception as e:
            self._logger.error(f"Error in main loop: {e}")
        finally:
            # Cleanup
            self.disable_output()
            self._logger.info("Main control loop ended")
    
    def _send_frame_data(self, frame: List[LaserPoint]) -> None:
        """Send frame data to the laser."""
        if not frame or not self.info:
            return
        
        # Convert points to bytes
        points_data = [point.to_bytes() for point in frame]
        
        while points_data and self.running:
            # Check buffer availability
            buffer_threshold = int(self.info.rx_buffer_size * 0.8)
            if self.remote_buf_free < buffer_threshold:
                # Wait for buffer space
                sleep_time = min(0.01, 100 / self.info.dac_rate if self.info.dac_rate > 0 else 0.01)
                time.sleep(sleep_time)
                self._estimate_buffer_recovery()
                continue
            
            # Determine chunk size
            max_chunk = min(MAX_SAMPLES_PER_UDP_PACKET, len(points_data), self.remote_buf_free)
            if max_chunk <= 0:
                time.sleep(0.001)
                continue
            
            # Create message
            header = bytes([
                Commands.SAMPLE_DATA,
                0x00,
                self._message_num % 256,
                self._frame_num % 256
            ])
            
            # Add points to message
            chunk_data = b''.join(points_data[:max_chunk])
            message = header + chunk_data
            
            # Send message
            try:
                self.data_sock.sendto(message, (self.addr, DATA_PORT))
                
                # Update counters
                self._message_num += 1
                self.remote_buf_free -= max_chunk
                points_data = points_data[max_chunk:]
                
            except OSError as e:
                self._logger.error(f"Failed to send frame data: {e}")
                break
        
        self._frame_num += 1


class LaserController:
    """
    Main controller class for discovering and managing LaserCube devices.
    
    Args:
        frame_generator: Function that returns a list of LaserPoint objects for each frame
    """
    
    def __init__(self, frame_generator: Callable[[], List[LaserPoint]]):
        self.frame_generator = frame_generator
        self.known_lasers: Dict[str, LaserCube] = {}
        self.running = False
        
        # Setup logging
        self._logger = logging.getLogger("LaserController")
        
        # Sockets
        self.cmd_sock: Optional[socket.socket] = None
        self.data_sock: Optional[socket.socket] = None
        
        # Threads
        self._scanner_thread: Optional[threading.Thread] = None
        self._receiver_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    def start(self) -> None:
        """Start the controller and begin laser discovery."""
        if self.running:
            return
        
        self.running = True
        self._stop_event.clear()
        
        # Setup sockets
        self._setup_sockets()
        
        # Start threads
        self._scanner_thread = threading.Thread(target=self._scanner_loop, daemon=True)
        self._scanner_thread.start()
        
        self._receiver_thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self._receiver_thread.start()
        
        self._logger.info("LaserController started")
    
    def stop(self) -> None:
        """Stop the controller and all connected lasers."""
        if not self.running:
            return
        
        self.running = False
        self._stop_event.set()
        
        # Stop all lasers
        for laser in self.known_lasers.values():
            laser.stop()
        
        # Close sockets
        if self.cmd_sock:
            self.cmd_sock.close()
        if self.data_sock:
            self.data_sock.close()
        
        self._logger.info("LaserController stopped")
    
    def get_connected_lasers(self) -> List[LaserCube]:
        """Get list of currently connected laser devices."""
        return [laser for laser in self.known_lasers.values() if laser.is_connected()]
    
    def discover_lasers(self) -> List[str]:
        """Manually trigger laser discovery and return found addresses."""
        if not self.cmd_sock:
            return []
        
        try:
            self.cmd_sock.sendto(bytes([Commands.GET_FULL_INFO]), ('255.255.255.255', CMD_PORT))
            # Give some time for responses
            time.sleep(1)
            return list(self.known_lasers.keys())
        except OSError as e:
            self._logger.error(f"Discovery error: {e}")
            return []
    
    def _setup_sockets(self) -> None:
        """Setup UDP sockets with proper configuration."""
        try:
            # Command socket
            self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.cmd_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self.cmd_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.cmd_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
            self.cmd_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
            self.cmd_sock.bind(('0.0.0.0', CMD_PORT))
            
            # Data socket
            self.data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.data_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.data_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 131072)
            self.data_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 131072)
            self.data_sock.bind(('0.0.0.0', DATA_PORT))
            
            self._logger.info("Sockets configured successfully")
            
        except OSError as e:
            self._logger.error(f"Failed to setup sockets: {e}")
            raise
    
    def _scanner_loop(self) -> None:
        """Periodically broadcast discovery messages."""
        while self.running and not self._stop_event.is_set():
            try:
                if self.cmd_sock:
                    self.cmd_sock.sendto(bytes([Commands.GET_FULL_INFO]), ('255.255.255.255', CMD_PORT))
                time.sleep(2)
            except OSError as e:
                if self.running:
                    self._logger.error(f"Scanner error: {e}")
    
    def _receiver_loop(self) -> None:
        """Receive and route messages from laser devices."""
        while self.running and not self._stop_event.is_set():
            try:
                # Check socket validity
                valid_sockets = []
                if self.cmd_sock and self.cmd_sock.fileno() != -1:
                    valid_sockets.append(self.cmd_sock)
                if self.data_sock and self.data_sock.fileno() != -1:
                    valid_sockets.append(self.data_sock)
                
                if not valid_sockets:
                    break
                
                # Wait for incoming data
                ready, _, _ = select.select(valid_sockets, [], [], 1.0)
                
                for sock in ready:
                    try:
                        msg, (addr, port) = sock.recvfrom(4096)
                        self._handle_message(addr, msg)
                    except OSError as e:
                        if self.running:
                            self._logger.error(f"Receive error: {e}")
                            
            except OSError as e:
                if self.running:
                    self._logger.error(f"Receiver loop error: {e}")
    
    def _handle_message(self, addr: str, msg: bytes) -> None:
        """Handle incoming message from a laser device."""
        if not msg:
            return
        
        # Create new laser if not known
        if addr not in self.known_lasers:
            if msg[0] == Commands.GET_FULL_INFO and len(msg) > 1:
                laser = LaserCube(addr, self.frame_generator, self.cmd_sock, self.data_sock)
                self.known_lasers[addr] = laser
                laser.start()
                self._logger.info(f"Discovered new laser at {addr}")
            else:
                return
        
        # Route message to appropriate laser
        laser = self.known_lasers[addr]
        was_connected = laser.is_connected()
        laser.process_message(msg)
        
        # Log new connections
        if not was_connected and laser.is_connected():
            self._logger.info(f"Connected to {laser.info}")
    
    def run(self) -> None:
        """Run the laser controller (legacy compatibility method)."""
        self.start()
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self._logger.info("Received interrupt signal")
        finally:
            self.stop()


def create_frame_from_bbox(bbox_data: Optional[Tuple[float, float, float, float]], 
                          screen_width: int, screen_height: int, 
                          power: int = 4095) -> List[LaserPoint]:
    """
    Generate laser frame data from bounding box coordinates.
    
    Args:
        bbox_data: Tuple of (x1, y1, x2, y2) bounding box coordinates, or None for center
        screen_width: Width of the screen/detection area
        screen_height: Height of the screen/detection area
        power: Laser power level (0-4095, default full power)
    
    Returns:
        List of LaserPoint objects representing the frame
    """
    if bbox_data is None:
        # Center point when no detection
        x, y = 2048, 2048
    else:
        x1, y1, x2, y2 = bbox_data
        # Calculate center coordinates
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        # Normalize to screen coordinates
        nx = center_x / screen_width
        ny = center_y / screen_height
        # Convert to laser coordinates (0-4095), flipped for laser coordinate system
        x = int((1 - nx) * 4095)
        y = int((1 - ny) * 4095)
    
    # Create frame with multiple points for visibility
    frame = []
    for _ in range(50):  # 50 points per frame for good visibility
        frame.append(LaserPoint(x, y, power, power, power))
    
    return frame