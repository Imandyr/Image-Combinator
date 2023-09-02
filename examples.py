from pathlib import Path
from natsort import natsorted

from defs import ImagesCombinatorPlt


source_path = Path("source_images") / "1"
output_path = Path("output") / "1"

# create new object using images from source
combinator = ImagesCombinatorPlt(source_path, lower_batch_index=0, upper_batch_index=7, sort=natsorted,
                                 batch_shape=(4, 5), logs=True)

# print its representation
print(combinator)

# get one processed image
print(combinator[4])

# take slice
print(combinator[2:5])

# iterate processed images
print([e for e in combinator])

# process and save all images
combinator.process_and_save_all(output_path, ".jpg")

# save processed images to gif file using ffmpeg
combinator.process_and_save_to_gif_ffmpeg(output_path, rewrite=True, frame_rate=1, delete_images_on_end=True)

# save processed images to gif file using pillow
combinator.process_and_save_to_gif_pillow(output_path / "output_2.gif", frame_rate=1, rewrite=True)

# read processed images as spectrogram and save created audio in wav file
combinator.process_and_save_to_wav(output_path / "output.wav", rewrite=True, duration=1)



