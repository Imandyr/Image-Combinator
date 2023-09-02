from typing import cast, Union, Iterable, Callable, Optional, NamedTuple, Tuple, Sized, List, Generator, Any, \
    overload, Iterator, Collection

from pathlib import Path
from abc import ABC, abstractmethod
import functools

import subprocess
from subprocess import DEVNULL, STDOUT

from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from PIL import Image
import numpy as np
from natsort import natsorted

from librosa import feature
import soundfile as sf


def split_list_to_batches(_list: list, batch_size: int) -> list:
    """Split given list to batches of batch_size."""
    list_of_batches = []
    batch = []
    for count, _path in enumerate(_list, start=1):
        if count % batch_size == 0:
            batch.append(_path)
            list_of_batches.append(batch)
            batch = []
        else:
            batch.append(_path)
    return list_of_batches


def convert_image_to_audio(img: Image.Image) -> np.ndarray:
    """Read image object as spectrogram and convert in to audio."""
    return feature.inverse.mel_to_audio(np.asarray(img.convert("L")).astype("float32"))


def progress_bar_for_generator_function(start_text: str, percents_per_iteration: Union[float, int]) \
        -> Callable[[Callable[..., Generator]], Callable[..., Generator]]:
    """
    Creates decorator function which adds text progress bar in terminal output for given generator function.
    :param start_text: Text in start of progress_bar.
    :param percents_per_iteration: Because number of iterations in generator objects is unknown,
    it's impossible to automatically calculate how many percents should be added on every iteration.
    So you must manually specify this value.
    :return: Decorator for generator function.
    """

    def decorator(func: Callable[..., Generator]):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in progress_bar_generator(func(*args, **kwargs), start_text, percents_per_iteration):
                yield i
        return wrapper
    return decorator


def progress_bar_generator(iterable: Union[Collection, Iterable], start_text: str,
                           percents_per_iteration: Optional[float]) -> Generator:
    """
    Takes iterable object and returns generator which iterate it and print progress of iteration in terminal output.
    :param iterable: Iterable object which will be iterated. Notice: if Iterable is also Sized, percents_per_iteration
    can be calculated automatically, so it can be passed as None.
    :param start_text: Text in start of progress_bar.
    :param percents_per_iteration: How many percents will be added on every iteration.
    Can be automatically calculated if iterable is Sized
    :return: Generator object which iterates iterable and print progress of it.
    """
    if percents_per_iteration is None:
        percents_per_iteration = 100 / len(iterable)
    count = 0
    for i in iterable:
        yield i
        count += percents_per_iteration
        print(f"\r{start_text}{count}%", end="")
    print("\n", end="")


class ImagesCombinator(ABC):
    """
    Base abstract class, instances of which combines every image from directory to batches of arbitrary size
    and saves every batch as one image file.
    Briefly speaking, it's combining images together.

    Notice: Because this class is abstract, it should only be used as base for subclasses, which should have
    concrete processing_generator() method implementation.
    """

    def __init__(self, source_directory: Union[Path, str],
                 lower_batch_index: Optional[int] = None,
                 upper_batch_index: Optional[int] = None,
                 batch_shape: Optional[Tuple[int, int]] = None,
                 sort: Optional[Callable[[Iterable[Path]], List[Path]]] = natsorted,
                 logs: bool = True) -> None:
        """
        This object combines every image from directory to batches of arbitrary size and saves every batch as one
        image file.
        Briefly speaking, it's combining images together.

        On initialization, this object only prepare images paths to processing.
        For just process and return combined images as python objects, use method .process().
        For process and then immediately save combined images to files, use method process_and_save().

        Notice 1: Images files names in directory is important, because this class process them in left to right order.
        Their position in loading order based on how pathlib.Path.iterdir() load images paths and how function
        specified in sort argument process them.

        Notice 2: Target images files will be loaded on first call of any method that uses them,
        which may take some time, but then they will be cached, so rest of calls will be faster.

        :param source_directory: Path to directory with images files. No other folders can be inside, only image files.
        All images should have a format supported by the Pillow Image class.

        :param lower_batch_index: Index of start images batch, on which object will start loading files.
        One batch of images is number of batch_shape[0] * batch_shape[1].

        :param upper_batch_index: Index of end images batch, on which object will end loading files.
        Final list of images to load is images_paths[lower_batch_index * batch_size:upper_batch_index * batch_size].

        :param batch_shape: If specified, number of rows and columns in which all images from one batch will be
        combined in one frame.
        batch_shape[0] is the number of rows and batch_shape[1] is the number of columns.

        :param sort: If specified, use a given sorting function on paths to images before loading them.
        If not, default sorting will be used.

        :param logs: if True, print progress of processing.

        """

        if isinstance(source_directory, Path) or isinstance(source_directory, str):
            self.source_directory = Path(source_directory)
            if not self.source_directory.exists():
                raise ValueError(f"Path '{self.source_directory}' is not exist.")
        else:
            raise ValueError(f"Argument 'source_directory' must have type pathlib.Path or str, "
                             f"not {source_directory.__class__.__name__}.")

        if sort is None:
            self.sort = None
            self.images_paths = list(self.source_directory.iterdir())
        else:
            if not isinstance(sort, Callable):
                raise ValueError(f"Argument 'sort' must be None or have type Callable, "
                                 f"not {cast(object, sort).__class__.__name__}.")
            self.sort = sort
            self.images_paths = self.sort(list(self.source_directory.iterdir()))
            if not isinstance(self.images_paths, list) or not all(isinstance(e, Path) for e in self.images_paths):
                raise ValueError(f"Function given to argument 'sort' is invalid.")

        if batch_shape is None:
            batch_shape = (1, 1)
        if not isinstance(batch_shape, tuple):
            raise ValueError(f"Argument 'batch_shape' must have type tuple, "
                             f"not {batch_shape.__class__.__name__}.")
        self.batch_shape = batch_shape
        self.batch_size = self.batch_shape[0] * self.batch_shape[1]

        if lower_batch_index is None:
            lower_batch_index = 0
        if not isinstance(lower_batch_index, int):
            raise ValueError(f"Argument 'lower_batch_index' must be None or have type 'int', "
                             f"not '{lower_batch_index.__class__.__name__}'")
        self.lower_batch_index = lower_batch_index

        if upper_batch_index is None:
            upper_batch_index = len(self.images_paths)
        if not isinstance(upper_batch_index, int):
            raise ValueError(f"Argument 'upper_batch_index' must be None or have type 'int', "
                             f"not '{upper_batch_index.__class__.__name__}'")
        self.upper_batch_index = upper_batch_index

        if self.lower_batch_index >= self.upper_batch_index:
            raise ValueError(f"lower_batch_index >= upper_batch_index.")

        if not isinstance(logs, bool):
            raise ValueError("Argument 'logs' must have type 'bool', "
                             f"not '{logs.__class__.__name__}'")
        self.logs = logs

        self.images_paths = self.images_paths[
                            self.lower_batch_index * self.batch_size:self.upper_batch_index * self.batch_size]
        self.images_paths_in_batches = np.asarray(split_list_to_batches(self.images_paths, self.batch_size),
                                                  dtype="str")
        self._images_list: List[Image.Image] = []
        self._audio_list: List[np.ndarray] = []

        if logs:
            self.processing_generator = progress_bar_for_generator_function(
                "Image processing: ", 100 / self.images_paths_in_batches.shape[0])(
                self.processing_generator)
            self.audio_from_images_generator = progress_bar_for_generator_function(
                "Converting images to audio: ", 100 / self.images_paths_in_batches.shape[0])(
                self.audio_from_images_generator)

    def __repr__(self) -> str:
        """Representation of class object."""
        return f"{self.__class__.__name__}(" \
               f"source_directory='{self.source_directory}', " \
               f"lower_batch_index={self.lower_batch_index}, " \
               f"upper_batch_index={self.upper_batch_index}, " \
               f"{'' if self.sort is None else f'sort={self.sort}, '}" \
               f"batch_shape={self.batch_shape}, " \
               f")"

    @overload
    def __getitem__(self, item: slice) -> "ImagesCombinator":
        f"""Get {self.__class__.__name__} object with combined images from input slice of batch indexes."""

    @overload
    def __getitem__(self, item: int) -> Image.Image:
        """Get combined image by it batch index."""

    def __getitem__(self, item):
        f"""Get {self.__class__.__name__} object with combined images from the input slice of the batch index, 
        or get combined image by it index."""
        if isinstance(item, slice):
            start = 0 if item.start is None else item.start
            stop = 0 if item.stop is None else item.stop
            return self.__class__(self.source_directory, self.lower_batch_index + start,
                                  self.lower_batch_index + stop, self.sort, self.batch_shape)
        if isinstance(item, int):
            return self.processed_images[item]
        else:
            raise ValueError(f"Argument 'item' expects object with type 'slice' or 'int'. "
                             f"Got '{item.__class__.__name__}' instead.")

    def __iter__(self) -> Iterator:
        """Returns iterator object for all combined images."""
        return iter(self.processed_images)

    def __len__(self) -> int:
        return len(self.processed_images)

    @abstractmethod
    def processing_generator(self) -> Generator[Image.Image, Any, None]:
        """Returns generator object which create and yield one processed batch image at time."""
        ...

    @property
    def processed_images(self) -> List[Image.Image]:
        """Returns a list with all processed images."""
        if len(self._images_list) == 0:
            self._images_list = list(self.processing_generator())
        return self._images_list.copy()

    def audio_from_images_generator(self) -> Generator[np.ndarray, Any, None]:
        """Returns generator object which read processed image as spectrogram and yield one generated audio at time."""
        for img in self.processed_images:
            yield feature.inverse.mel_to_audio(np.asarray(img.convert("L")).astype("float32"))

    @property
    def audio_from_images(self) -> List[np.ndarray]:
        """Returns a list with all audio created from images."""
        if len(self._audio_list) == 0:
            self._audio_list = list(self.audio_from_images_generator())
        return self._audio_list.copy()

    def process_and_save_all(self, fp: Union[Path, str], _format: str) -> None:
        """Process all images and save result as file in given directory and in given format."""
        fp = Path(fp)
        digits_number = len(str(self.images_paths_in_batches.shape[0]))
        for count, img in enumerate(self.processed_images):
            image_number = str(count).zfill(digits_number)
            img.save(fp=Path(fp) / f"img_{image_number}{_format}")

    def process_and_save_to_gif_ffmpeg(self, fp: Union[Path, str], rewrite: bool = False, frame_rate: int = 1,
                                       command: Optional[str] = None,
                                       use_existing_images: bool = False, delete_images_on_end: bool = True) -> None:
        """
        This method is creates gif file from all combined images using ffmpeg.
        If there is not ffmpeg installed - method can't work.

        For that purpose method takes a directory file path (if rewrite == False, this directory should be empty),
        temporally writes as files all combined images to it, unite them to single gif file using ffmpeg command
        and then deletes all temporally created images, so only gif file remains.

        :param fp: File path to target directory. It's very recommended to use empty directory,
        because otherwise some of the original files may be lost.
        :param rewrite: If True, allows method to rewrites any files inside given directory.
        :param frame_rate: Frame rate of created GIF file.
        :param command: Optional parameter for custom ffmpeg command. If specified,
        replace default command used in this method. Should be used on your own caution.
        FFMPEG command example: "ffmpeg -f image2 -framerate 10 -i "img_%02d.png" output.gif -y".
        :param use_existing_images: Use already existing images from fp directory in creation of gif file.
        Should be used on your own caution, because images used in this method should have very specific
        name and format. If there is no images already, method firstly creates them anyway.
        :param delete_images_on_end: Delete or not images after creation of gif file.
        :return: None.
        """
        fp = Path(fp)
        fp_is_empty = len(list(fp.iterdir())) == 0
        if not rewrite and not fp_is_empty:
            raise ValueError(f"Directory specified in argument 'fp' is not empty, "
                             f"but value of argument 'rewrite' is {rewrite}.")

        if not fp.is_dir():
            raise ValueError("Path specified in 'fp' argument is ether not a directory or didn't exist.")

        if not use_existing_images or fp_is_empty:
            self.process_and_save_all(fp, ".png")

        if command is None:
            digits_number = len(str(self.images_paths_in_batches.shape[0]))
            command = f'ffmpeg -framerate {frame_rate} -i "{fp / "img_"}%{str(digits_number).zfill(2)}d.png" ' \
                      f'"{fp / "output"}.gif" -y'
        subprocess.Popen(command, shell=True, stdout=DEVNULL, stderr=STDOUT)

        if delete_images_on_end:
            for file in fp.iterdir():
                if file.is_file() and file.suffix == ".png" and file.name.find("img_") == 0:
                    file.unlink()

    def process_and_save_to_gif_pillow(self, fp: Union[Path, str], frame_rate: int = 1, rewrite: bool = False) -> None:
        """
        This method is creates and save gif file from all combined images using Pillow.

        :param fp: File path and name of gif file which will be created.
        :param frame_rate: Frame rate of gif file in seconds.
        :param rewrite: Allows to rewrite file on given path if its already exist.
        :return: None.
        """
        fp = Path(fp)
        if fp.suffix == "":
            fp = Path(str(fp) + ".gif")
        if not fp.suffix == ".gif":
            raise ValueError("Argument 'fp' must have '.gif' suffix.")
        if fp.exists() and not rewrite:
            raise ValueError(f"File on given file path: '{fp}' is already exist, but argument 'rewrite' is False.")

        images_list = self.processed_images
        duration = 1000 / frame_rate
        images_list[0].save(fp=fp, save_all=True, append_images=images_list[1:], duration=duration, loop=0)

    def process_and_save_to_wav(self, fp: Union[Path, str], rewrite: bool = False,
                                duration: Union[float, int, None] = None, sr: Optional[int] = None) -> None:
        """
        Method for converting all processed images to audio via reading them as spectrogram and saving them to .wav file.
         Why? Because I can.
        :param fp: Path to output .wav file.
        :param rewrite: Allows to rewrite file on given path if its already exist.
        :param duration: Duration of one audio piece created from one combined image
         (thus, audio file length == duration * processed images count). If specified, argument 'sr' must be None,
         because in this case sample rate will be automatically adjusted to fit in given duration.
        :param sr: Sample Rate of output audio. If specified, argument 'duration' must be None.
        :return: None
        """
        fp = Path(fp)
        if fp.suffix == "":
            fp = Path(str(fp) + ".wav")
        if not fp.suffix == ".wav":
            raise ValueError("Argument 'fp' must have '.wav' suffix.")
        if fp.exists() and not rewrite:
            raise ValueError(f"File on given file path: '{fp}' is already exist, but argument 'rewrite' is False.")
        if duration is not None and sr is not None:
            raise ValueError("Arguments 'duration' and 'sr' cannot be specified together - "
                             " use only one of them instead.")
        if duration is None and sr is None:
            duration = 1

        audio = np.asarray(self.audio_from_images)
        audio_shape = audio.shape
        audio = audio.reshape((audio_shape[0] * audio_shape[1]))

        if sr is not None:
            sf.write(fp, audio, sr)
        else:
            sr = audio_shape[1] // duration
            sf.write(fp, audio, int(sr))


class ImagesCombinatorPlt(ImagesCombinator):
    """
    Class, instances of which combines every image from directory to batches of arbitrary size
    and saves every batch as one image file.
    Briefly speaking, it's combining images together.

    This version is uses Matplotlib for image combination.

    Notice: Original size of images most likely to lose, because combination happen via matplotlib Figure class
    with which I don't know how to do size lossless image compilation. so it is as it is.
    """

    def processing_generator(self) -> Generator[Image.Image, Any, None]:
        """
        Returns generator object which create and yield one batch image at time,
        using Matplotlib for image combination.
        All images yielding as Pillow.Image.Image instances with RGB color mode.
        """
        for batch in self.images_paths_in_batches:
            images_batch = [Image.open(_path) for _path in batch]
            figure = plt.Figure(figsize=(self.batch_shape[1], self.batch_shape[0]))
            num_elements = 0
            for row in range(self.batch_shape[0]):
                for column in range(self.batch_shape[1]):
                    ax = figure.add_subplot(self.batch_shape[0], self.batch_shape[1], num_elements + 1, )
                    ax.imshow(images_batch[num_elements])
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    num_elements += 1
            figure.subplots_adjust(wspace=0, hspace=0, bottom=0, left=0, right=1, top=1)
            figure_canvas = FigureCanvasAgg(figure)
            figure_canvas.draw()
            img = Image.frombuffer(mode="RGBA", data=figure_canvas.buffer_rgba(),
                                   size=cast(tuple, figure_canvas.get_width_height()))
            yield img.convert("RGB")


