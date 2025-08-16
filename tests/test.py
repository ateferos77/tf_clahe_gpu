from clahe import CLAHEConfig
import numpy as np

config = CLAHEConfig()

imgs = np.random.randint(0, 256, (1000, 224, 224), dtype=np.uint8)

print(config.apply(imgs).shape)
