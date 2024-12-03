# fmt: off
from aie.helpers.taplib import TensorAccessPattern, TensorAccessSequence

tap = TensorAccessPattern((4, 8), offset=0, sizes=[2, 3], strides=[16, 2])
tap.visualize(plot_access_count=True)

from aie.helpers.taplib import TensorAccessPattern, TensorAccessSequence
tap0 = TensorAccessPattern((4, 8), offset=0, sizes=[2, 3], strides=[16, 2])
tap1 = TensorAccessPattern((4, 8), offset=1, sizes=[2, 3], strides=[16, 2])
taps = TensorAccessSequence.from_tiles([tap0, tap1])
taps.visualize()


from aie.helpers.taplib import TensorAccessPattern, TensorAccessSequence
tap0 = TensorAccessPattern((4, 8), offset=0, sizes=[2, 3], strides=[16, 2])
tap1 = TensorAccessPattern((4, 8), offset=1, sizes=[2, 3], strides=[16, 2])
taps = TensorAccessSequence.from_tiles([tap0, tap1])
anim = taps.animate()

# View in notebook
HTML(anim.to_jshtml())

# Save to file
anim.save("myfile.gif")

from aie.helpers.taplib import TensorTiler2D
taps = TensorTiler2D.step_tiler(
    (4, 8), 
    (1, 1), 
    tile_group_repeats=[2, 3], 
    tile_group_steps=[2, 2], 
    allow_partial=True)
anim = taps.animate()

# View in notebook
HTML(anim.to_jshtml())

# Save to file
anim.save("myfile.gif")
# fmt: on