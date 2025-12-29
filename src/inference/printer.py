#!/usr/bin/env python3
"""
ESC/POS Printer Driver for SushiVoice
Thermal label printing for Marka and compatible printers
"""

import logging
from typing import Optional

try:
    from escpos.printer import Usb, Serial, Network, Dummy
    ESCPOS_AVAILABLE = True
except ImportError:
    ESCPOS_AVAILABLE = False
    logging.warning("python-escpos not installed. Printer functionality will be disabled.")


class SushiPrinter:
    """Thermal printer interface for label printing"""
    
    def __init__(
        self,
        printer_type: str = 'usb',
        vendor_id: Optional[int] = None,
        product_id: Optional[int] = None,
        serial_port: Optional[str] = None,
        ip_address: Optional[str] = None,
        port: int = 9100,
        dummy_mode: bool = False
    ):
        """
        Initialize printer
        
        Args:
            printer_type: Type of printer connection ('usb', 'serial', 'network', 'dummy')
            vendor_id: USB vendor ID (for USB printers)
            product_id: USB product ID (for USB printers)
            serial_port: Serial port path (for serial printers)
            ip_address: IP address (for network printers)
            port: Network port (default 9100)
            dummy_mode: Use dummy printer for testing
        """
        self.printer_type = printer_type
        self.printer = None
        self.dummy_mode = dummy_mode
        
        if not ESCPOS_AVAILABLE:
            logging.warning("Printer disabled: python-escpos not installed")
            self.dummy_mode = True
        
        # Initialize printer based on type
        try:
            if self.dummy_mode or printer_type == 'dummy':
                self.printer = Dummy()
                logging.info("Using dummy printer (test mode)")
            elif printer_type == 'usb':
                if vendor_id and product_id:
                    self.printer = Usb(vendor_id, product_id)
                    logging.info(f"Connected to USB printer {vendor_id:04x}:{product_id:04x}")
                else:
                    raise ValueError("USB printer requires vendor_id and product_id")
            elif printer_type == 'serial':
                if serial_port:
                    self.printer = Serial(serial_port)
                    logging.info(f"Connected to serial printer on {serial_port}")
                else:
                    raise ValueError("Serial printer requires serial_port")
            elif printer_type == 'network':
                if ip_address:
                    self.printer = Network(ip_address, port)
                    logging.info(f"Connected to network printer at {ip_address}:{port}")
                else:
                    raise ValueError("Network printer requires ip_address")
            else:
                raise ValueError(f"Unknown printer type: {printer_type}")
        except Exception as e:
            logging.error(f"Failed to initialize printer: {e}")
            logging.info("Falling back to dummy printer")
            self.printer = Dummy() if ESCPOS_AVAILABLE else None
            self.dummy_mode = True
    
    def print_label(self, item_name: str, quantity: int = 1):
        """
        Print labels for sushi items
        
        Args:
            item_name: Name of the sushi item
            quantity: Number of labels to print
        """
        if self.printer is None:
            # Console-friendly output: show each label
            print("\n📄 PRINTING LABELS:")
            print("=" * 40)
            for i in range(quantity):
                print(f"  {item_name}")
            print("=" * 40)
            print(f"✅ {quantity} label(s) printed\n")
            return
        
        # Show console output for dummy mode
        if self.dummy_mode:
            print("\n📄 PRINTING LABELS:")
            print("=" * 40)
            for i in range(quantity):
                print(f"  {item_name}")
            print("=" * 40)
            print(f"✅ {quantity} label(s) printed\n")
        
        try:
            for i in range(quantity):
                # Set formatting
                self.printer.set(
                    align='center',
                    font='a',
                    width=2,
                    height=2,
                    bold=True
                )
                
                # Print item name
                self.printer.text(f"{item_name}\n")
                
                # Add separator
                self.printer.set(width=1, height=1, bold=False)
                self.printer.text("-" * 20 + "\n")
                
                # Optional: Add label number if multiple
                if quantity > 1:
                    self.printer.text(f"#{i+1} of {quantity}\n")
                
                # Cut paper
                self.printer.cut()
                
                logging.info(f"Printed label {i+1}/{quantity} for {item_name}")
        
        except Exception as e:
            logging.error(f"Print error: {e}")
            # Fallback to console print
            print(f"[PRINT ERROR] {quantity}x {item_name} - {e}")
    
    def print_simple_label(self, item_name: str):
        """
        Print a single simple label (just item name)
        
        Args:
            item_name: Name of the sushi item
        """
        if self.printer is None:
            print(f"[DUMMY PRINT] {item_name}")
            return
        
        try:
            self.printer.set(align='center', font='a', width=2, height=2)
            self.printer.text(f"{item_name}\n")
            self.printer.cut()
            logging.info(f"Printed label for {item_name}")
        except Exception as e:
            logging.error(f"Print error: {e}")
            print(f"[PRINT ERROR] {item_name} - {e}")
    
    def test_print(self):
        """Print a test label"""
        if self.printer is None:
            print("[DUMMY TEST PRINT]")
            return
        
        try:
            self.printer.set(align='center')
            self.printer.text("SushiVoice Test\n")
            self.printer.text("================\n")
            self.printer.set(width=2, height=2)
            self.printer.text("TEST LABEL\n")
            self.printer.set(width=1, height=1)
            self.printer.text("If you can read this,\n")
            self.printer.text("the printer works!\n")
            self.printer.cut()
            logging.info("Test print successful")
            return True
        except Exception as e:
            logging.error(f"Test print failed: {e}")
            return False
    
    def close(self):
        """Close printer connection"""
        if self.printer and not self.dummy_mode:
            try:
                self.printer.close()
                logging.info("Printer connection closed")
            except:
                pass


def find_usb_printers():
    """
    Find connected USB printers
    
    Returns:
        List of (vendor_id, product_id) tuples
    """
    if not ESCPOS_AVAILABLE:
        print("python-escpos not installed")
        return []
    
    try:
        import usb.core
        
        # Find all USB devices
        devices = usb.core.find(find_all=True)
        printers = []
        
        for device in devices:
            # ESC/POS printers are usually in class 7 (printer)
            if device.bDeviceClass == 7 or any(
                iface.bInterfaceClass == 7 for cfg in device for iface in cfg
            ):
                printers.append((device.idVendor, device.idProduct))
                print(f"Found printer: {device.idVendor:04x}:{device.idProduct:04x}")
        
        return printers
    except Exception as e:
        logging.error(f"Error finding USB printers: {e}")
        return []


if __name__ == '__main__':
    # Test printer
    print("Testing Sushi Printer...\n")
    
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    
    # Try to find USB printers
    print("Searching for USB printers...")
    usb_printers = find_usb_printers()
    
    if usb_printers:
        print(f"\nFound {len(usb_printers)} printer(s)")
        print("To use a printer, update the vendor_id and product_id in the code")
    else:
        print("No USB printers found")
    
    # Use dummy printer for testing
    print("\nTesting with dummy printer...")
    printer = SushiPrinter(printer_type='dummy', dummy_mode=True)
    
    # Test prints
    print("\n1. Test print:")
    printer.test_print()
    
    print("\n2. Simple label:")
    printer.print_simple_label("Chicken Teriyaki")
    
    print("\n3. Multiple labels:")
    printer.print_label("California Roll", quantity=3)
    
    print("\n4. Single label:")
    printer.print_label("Salmon Nigiri", quantity=1)
    
    # Close
    printer.close()
    
    print("\n✅ Printer test complete!")
    print("\n💡 To use a real printer:")
    print("   1. Find your printer VID:PID using: lsusb (Linux/Mac) or Device Manager (Windows)")
    print("   2. Initialize with: SushiPrinter('usb', vendor_id=0xVVVV, product_id=0xPPPP)")
    print("   3. For Marka printer, check documentation for correct VID:PID")
