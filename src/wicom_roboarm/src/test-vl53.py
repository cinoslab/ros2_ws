import time
import board
import busio
import adafruit_tca9548a
import adafruit_vl53l1x
import adafruit_vl53l0x

# Create I2C bus and TCA9548A multiplexer
i2c = busio.I2C(board.SCL, board.SDA)
tca = adafruit_tca9548a.TCA9548A(i2c)

# Initialize VL53L1X (on channel 1)
print("Initializing VL53L1X on channel 1...")
try:
    vl53l1x = adafruit_vl53l1x.VL53L1X(tca[1])
    vl53l1x.start_ranging()
    print("VL53L1X initialized!")
except Exception as e:
    print(f"VL53L1X init error: {e}")
    vl53l1x = None

# Initialize VL53L0X (on channel 0)
print("Initializing VL53L0X on channel 0...")
try:
    vl53l0x = adafruit_vl53l0x.VL53L0X(tca[0])
    print("VL53L0X initialized!")
except Exception as e:
    print(f"VL53L0X init error: {e}")
    vl53l0x = None

# Main loop
while True:
    try:
        if vl53l1x:
            if vl53l1x.data_ready:
                distance1 = vl53l1x.distance
            else:
                distance1 = None
        else:
            distance1 = None
            
        if vl53l0x:
            distance2 = vl53l0x.range
        else:
            distance2 = None

        print(f"VL53L1X (channel 1): {distance1} mm | VL53L0X (channel 0): {distance2} mm")

    except Exception as e:
        print(f"Reading error: {e}")

    time.sleep(0.5)