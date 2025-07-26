"""
LaserCube UDP Controller

A clean, well-documented Python library for controlling Wicked Lasers LaserCube devices
over UDP network protocol. Supports discovery, connection, and real-time point streaming.

Based on the LaserDock protocol:
https://github.com/Wickedlasers/libLaserdockCore/blob/master/3rdparty/laserdocklib/src/LaserDockNetworkDevice.cpp
"""

import collections
import logging
import select
import socket
import struct
import threading
import time
from typing import List, Tuple, Optional, Callable, NamedTuple
from dataclasses import dataclass


# UDP Ports
ALIVE_PORT = 45456    # For alive/ping messages
CMD_PORT = 45457      # For commands (info, enable/disable, settings)
DATA_PORT = 45458     # For actual laser point data

# Protocol Commands
class Commands:
    GET_FULL_INFO = 0x77
    ENABLE_BUFFER_SIZE_RESPONSE_ON_DATA = 0x78
    SET_OUTPUT = 0x80
    GET_RINGBUFFER_EMPTY_SAMPLE_COUNT = 0x8a
    SAMPLE_DATA = 0xa9


@dataclass
class LaserInfo:
    """Information about a connected LaserCube device."""
    model_name: str
    fw_major: int
    fw_minor: int
    output_enabled: bool
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


class LaserPoint:
    """Represents a single laser point with position and color."""
    def __init__(self, x: int, y: int, r: int, g: int, b: int):
        # Clamp values to valid range (0-4095)
        self.x = max(0, min(4095, x))
        self.y = max(0, min(4095, y))
        self.r = max(0, min(4095, r))
        self.g = max(0, min(4095, g))
        self.b = max(0, min(4095, b))
    
    def to_bytes(self) -> bytes:
        """Convert point to binary format for transmission."""
        return struct.pack('<HHHHH', self.x, self.y, self.r, self.g, self.b)


class LaserCube:
    """
    Represents a single LaserCube device and handles communication with it.
    
    Args:
        addr: IP address of the LaserCube
        frame_generator: Function that returns a list of LaserPoint objects for each frame
    """
    
    def __init__(self, addr: str, frame_generator: Callable[[], List[LaserPoint]]):
        self.addr = addr
        self.frame_generator = frame_generator
        self.info: Optional[LaserInfo] = None
        self.remote_buf_free = 0
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._logger = logging.getLogger(f"LaserCube[{addr}]")
        
        # Create data socket for this specific laser
        self.data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.data_sock.bind(('0.0.0.0', 0))  # Bind to any available port
    
    def start(self) -> None:
        """Start the laser control thread."""
        if not self.running:
            self.running = True
            self._thread = threading.Thread(target=self._main_loop, daemon=True)
            self._thread.start()
            self._logger.info("Started laser control thread")
    
    def stop(self) -> None:
        """Stop the laser control thread and disable output."""
        if self.running:
            self.running = False
            if self._thread:
                self._thread.join(timeout=2.0)
            self._logger.info("Stopped laser control thread")
    
    def process_message(self, msg: bytes) -> None:
        """Process incoming message from the laser."""
        if not msg:
            return
            
        cmd = msg[0]
        
        if cmd == Commands.GET_FULL_INFO:
            self._parse_info_message(msg)
        elif cmd == Commands.ENABLE_BUFFER_SIZE_RESPONSE_ON_DATA:
            self._logger.debug("Buffer size response enabled acknowledged")
        elif cmd == Commands.GET_RINGBUFFER_EMPTY_SAMPLE_COUNT:
            if len(msg) >= 4:
                self.remote_buf_free = struct.unpack('<xxH', msg)[0]
                self._logger.debug(f"Buffer free space: {self.remote_buf_free}")
    
    def _parse_info_message(self, msg: bytes) -> None:
        """Parse the full info message from the laser."""
        try:
            # Unpack structured data
            fields = struct.unpack('<xxBB?5xIIxHHBBB11xB26x', msg)
            serial_bytes = struct.unpack('6B', msg[26:32])
            ip_bytes = struct.unpack('4B', msg[32:36])
            name = msg[38:].split(b'\0', 1)[0].decode('utf-8', errors='ignore')
            
            # Create info object
            info = LaserInfo(
                model_name=name,
                fw_major=fields[0],
                fw_minor=fields[1],
                output_enabled=fields[2],
                dac_rate=fields[3],
                max_dac_rate=fields[4],
                rx_buffer_free=fields[5],
                rx_buffer_size=fields[6],
                battery_percent=fields[7],
                temperature=fields[8],
                connection_type=fields[9],
                model_number=fields[10],
                serial_number=':'.join(f'{b:02x}' for b in serial_bytes),
                ip_addr='.'.join(str(b) for b in ip_bytes)
            )
            
            if info != self.info:
                self.info = info
                self.remote_buf_free = info.rx_buffer_free
                self._logger.info(f"Updated laser info: {info.model_name} v{info.fw_major}.{info.fw_minor}")
                
        except (struct.error, IndexError) as e:
            self._logger.error(f"Failed to parse info message: {e}")
    
    def _send_command(self, cmd: List[int]) -> None:
        """Send a command to the laser (sent twice for reliability)."""
        cmd_data = bytes(cmd)
        try:
            # Send twice for reliability
            for _ in range(2):
                self.data_sock.sendto(cmd_data, (self.addr, CMD_PORT))
        except OSError as e:
            self._logger.error(f"Failed to send command {cmd}: {e}")
    
    def _main_loop(self) -> None:
        """Main loop for sending laser data."""
        message_num = 0
        frame_num = 0
        
        # Enable output and buffer size responses
        self._send_command([Commands.ENABLE_BUFFER_SIZE_RESPONSE_ON_DATA, 0x1])
        self._send_command([Commands.SET_OUTPUT, 0x1])
        
        try:
            while self.running:
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
                
                # Send frame in chunks
                points_data = [point.to_bytes() for point in current_frame]
                
                while points_data:
                    # Wait if buffer is getting full
                    if self.info and self.remote_buf_free < (self.info.rx_buffer_size * 0.8):
                        sleep_time = 100 / self.info.dac_rate if self.info.dac_rate > 0 else 0.001
                        time.sleep(sleep_time)
                        self.remote_buf_free += 100  # Estimate buffer recovery
                    
                    # Create message header
                    msg = bytes([Commands.SAMPLE_DATA, 0x00, message_num % 0xff, frame_num % 0xff])
                    
                    # Add up to 140 points per message (keeps under MTU)
                    chunk_size = min(140, len(points_data))
                    for i in range(chunk_size):
                        msg += points_data[i]
                        self.remote_buf_free -= 1
                    
                    # Send message
                    try:
                        self.data_sock.sendto(msg, (self.addr, DATA_PORT))
                        message_num += 1
                        points_data = points_data[chunk_size:]
                    except OSError as e:
                        self._logger.error(f"Failed to send data: {e}")
                        break
                
                frame_num += 1
                
        finally:
            # Cleanup: disable output and buffer responses
            self._send_command([Commands.ENABLE_BUFFER_SIZE_RESPONSE_ON_DATA, 0x0])
            self._send_command([Commands.SET_OUTPUT, 0x0])
            self._logger.info("Laser output disabled")


class LaserCubeController:
    """
    Main controller class for discovering and managing LaserCube devices.
    
    Args:
        frame_generator: Function that returns a list of LaserPoint objects for each frame
    """
    
    def __init__(self, frame_generator: Callable[[], List[LaserPoint]]):
        self.frame_generator = frame_generator
        self.known_lasers: dict[str, LaserCube] = {}
        self.running = False
        
        # Setup logging
        self._logger = logging.getLogger("LaserCubeController")
        
        # Create sockets
        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.cmd_sock.bind(('0.0.0.0', CMD_PORT))
        
        self.data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.data_sock.bind(('0.0.0.0', DATA_PORT))
        
        # Threads
        self._scanner_thread: Optional[threading.Thread] = None
        self._receiver_thread: Optional[threading.Thread] = None
    
    def start(self) -> None:
        """Start the controller and begin laser discovery."""
        if self.running:
            return
            
        self.running = True
        
        # Start scanner thread
        self._scanner_thread = threading.Thread(target=self._scanner_loop, daemon=True)
        self._scanner_thread.start()
        
        # Start receiver thread
        self._receiver_thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self._receiver_thread.start()
        
        self._logger.info("LaserCube controller started")
    
    def stop(self) -> None:
        """Stop the controller and all connected lasers."""
        if not self.running:
            return
            
        self.running = False
        
        # Stop all lasers
        for laser in self.known_lasers.values():
            laser.stop()
        
        # Close sockets
        self.cmd_sock.close()
        self.data_sock.close()
        
        self._logger.info("LaserCube controller stopped")
    
    def get_connected_lasers(self) -> List[LaserCube]:
        """Get list of currently connected laser devices."""
        return [laser for laser in self.known_lasers.values() if laser.info is not None]
    
    def _scanner_loop(self) -> None:
        """Periodically broadcast discovery messages."""
        while self.running:
            try:
                self.cmd_sock.sendto(bytes([Commands.GET_FULL_INFO]), ('255.255.255.255', CMD_PORT))
                time.sleep(1)
            except OSError as e:
                if self.running:  # Only log if we're still supposed to be running
                    self._logger.error(f"Scanner error: {e}")
    
    def _receiver_loop(self) -> None:
        """Receive and route messages from laser devices."""
        while self.running:
            try:
                ready, _, _ = select.select([self.cmd_sock, self.data_sock], [], [], 1.0)
                
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
        if addr not in self.known_lasers:
            # New laser discovered
            if msg[0] == Commands.GET_FULL_INFO and len(msg) > 1:
                laser = LaserCube(addr, self.frame_generator)
                self.known_lasers[addr] = laser
                laser.start()
                self._logger.info(f"Discovered new laser at {addr}")
            else:
                return
        
        # Route message to appropriate laser
        laser = self.known_lasers[addr]
        laser.process_message(msg)


# Fixed power level
FIXED_POWER = 1024  # Fixed power level (50% of 4095)

# Example usage
def create_mouse_tracking_frame() -> List[LaserPoint]:
    """Example frame generator that tracks mouse position."""
    try:
        import pyautogui
        
        screen_x, screen_y = pyautogui.size()
        mouse_x, mouse_y = pyautogui.position()
        
        # Convert mouse position to laser coordinates (0-4095)
        laser_x = 4095 - int((mouse_x * 4095) / screen_x)
        laser_y = 4095 - int((mouse_y * 4095) / screen_y)
        
        # Create a frame with multiple points at the mouse position using fixed power
        frame = []
        for _ in range(256):
            frame.append(LaserPoint(laser_x, laser_y, FIXED_POWER, FIXED_POWER, FIXED_POWER))
        
        return frame
        
    except ImportError:
        # Fallback if pyautogui not available
        return [LaserPoint(2048, 2048, FIXED_POWER, FIXED_POWER, FIXED_POWER) for _ in range(256)]
    except Exception as e:
        logging.error(f"Error in frame generator: {e}")
        return []


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start controller
    controller = LaserCubeController(create_mouse_tracking_frame)
    controller.start()
    
    try:
        print("LaserCube controller starting...")
        print("Move your mouse to control the laser position.")
        print(f"Fixed power level: {FIXED_POWER}/4095 (50%)")
        
        # Wait for laser connection
        print("Waiting for laser connection...")
        while True:
            lasers = controller.get_connected_lasers()
            if lasers:
                for laser in lasers:
                    if laser.info:
                        print(f"Connected: {laser.info.model_name} at {laser.addr}")
                        print(f"Battery: {laser.info.battery_percent}%")
                        break
                break
            time.sleep(1)
        
        print("\nLaser ready! Move mouse to control position.")
        print("Press Ctrl+C to quit.")
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
        controller.stop()
        print("Stopped.")