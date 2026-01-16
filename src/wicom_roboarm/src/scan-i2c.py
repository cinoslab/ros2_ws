import time
import board
import busio
import adafruit_tca9548a

# Create I2C bus
i2c = busio.I2C(board.SCL, board.SDA)

# Initialize multiplexer
tca = adafruit_tca9548a.TCA9548A(i2c)

print("Scanning all TCA9548A channels...")

for channel in range(8):
    try:
        # Try to scan this channel
        print(f"\nScanning channel {channel}:")
        tca[channel].try_lock()
        addresses = tca[channel].scan()
        tca[channel].unlock()
        
        if addresses:
            print(f"  Found devices at: {[hex(addr) for addr in addresses]}")
        else:
            print("  No devices found")
            
    except Exception as e:
        print(f"  Error scanning channel {channel}: {e}")

print("\nScanning main I2C bus...")
i2c.try_lock()
addresses = i2c.scan()
i2c.unlock()
print(f"Main bus devices: {[hex(addr) for addr in addresses]}")