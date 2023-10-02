def funct_convert_from_mp3_to_wov(var_audio_folder_input_path, var_audio_folder_output_path):

  """
  only google colab and google drive!!!

  var_audio_folder_input_path: specify the path to the folder with mp3 files
  var_audio_folder_output_path: specify the path to the folder where will be wav files
  return: get mp3 files from google drive and covert them to wav and put in the google drive

  # need for work:
  !pip install pydub

  # start in google colab:
  from google.colab import drive
  drive.mount('/content/drive')

  example for call a function in a program:

  # declare a variables with pathes to folders google drive with files
  var_audio_folder_input_path = '/content/drive/MyDrive/06 App ausk/p_001_audio/p_001_folder_audio_for_train/p_001_train_normal_breath'
  var_audio_folder_output_path = '/content/drive/MyDrive/06 App ausk/p_001_audio/p_001_folder_audio_for_train/p_001_train_normal_breath_wav'

  # call a function for convert files from mp3 to wav
  funct_convert_from_mp3_to_wov(var_audio_folder_input_path = var_audio_folder_input_path,
   var_audio_folder_output_path = var_audio_folder_output_path)
  """

  # do imports
  import os
  import shutil
  from pydub import AudioSegment

  # variable temp warhous
  var_content_str = '/content/'


  # Use os.listdir() to get a list of all files and directories in the folder
  var_name_audio_list = os.listdir(var_audio_folder_input_path)

    # start cycle for
  for var_name_audio_file in var_name_audio_list:
    # get path for input file + name file str
    var_path_input_file_str = var_audio_folder_input_path + '/' + var_name_audio_file

    # Load the MP3 file using pydub
    var_object_audio = AudioSegment.from_mp3(var_path_input_file_str)

    # del .mp3 from name file
    var_name_output_file = var_name_audio_file[:-4]

    # add to name audio file extension .wav
    var_name_output_file_and_wav = var_name_output_file + '.wav'

    # export the audio as WAV
    var_object_audio.export(var_name_output_file_and_wav, format="wav")

    # create puth to output folder
    var_path_in_colab_str = var_content_str + var_name_output_file_and_wav

    # write all wav files to google drive
    shutil.copy(var_path_in_colab_str, var_audio_folder_output_path)

    print('files_wrote to adress: ' + var_audio_folder_output_path + '/' + var_name_output_file_and_wav)

    # putch to folder input
    var_puth_folder_input_str = '/content/drive/MyDrive/06 App ausk/p_001_audio/p_001_folder_audio_for_train/p_001_train_normal_breath_wav'

    # putch to folder output
    var_puth_folder_output_str = '/content/sample_data'

def func_convert_wav_44100_to_16000_grz(var_puth_folder_input_str, var_puth_folder_output_str):
  import os
  import librosa
  import soundfile as sf

  # get list file name
  # Use os.listdir() to get a list of all files and directories in the folder
  var_name_audio_list = os.listdir(var_puth_folder_input_str)

  # create cycle for
  for var_name_one_audio_fail_str in var_name_audio_list:
    # get one file
    # create variable with puth to file (folder + file)
    var_full_puth_to_file = var_puth_folder_input_str + '/' + var_name_one_audio_fail_str

    # convert file
    audio, sr = librosa.load(var_full_puth_to_file, sr=16000)  # Resample to 16,000 Hz

    # create variable with puth to file output (folder + file)
    var_full_puth_to_file_output_str = var_puth_folder_output_str + '/' + var_name_one_audio_fail_str

    # put in one file to folder
    # Save the resampled audio as a WAV file
    sf.write(var_full_puth_to_file_output_str, audio, sr)

def func_preprocesing_audio_to_spectrogram(var_path_one_file_str, test_var):
  '''convert wav file to 16000 Hz and build spectrogram one file
  # install libraries
  !pip install tensorflow_io
  # Conect to google drive
  from google.colab import drive
  drive.mount('/content/drive')
  # Import packages
  import os
  from IPython import display
  import matplotlib.pyplot as plt
  import numpy as np
  import pandas as pd
  import tensorflow as tf
  import tensorflow_hub as hub
  import tensorflow_io as tfio
  # Put the path from van file
  var_path_train_normal_breath_wav = '/content/drive/MyDrive/P_001/all_audio_files/p_001_train_normal_breath_wav/1_01Vesicular.wav'
  var_path_train_not_normal_breath_wav = '/content/drive/MyDrive/P_001/all_audio_files/p_001_train_not_normal_breath_wav/1_02DimVesicular.wav'
  # Use the path as variable in function
  test_var = 32000
  spectr_1 = func_preprocesing_audio_to_spectrogram(var_path_one_file_str = var_path_train_normal_breath_wav, test_var = test_var)
  spectr_2 = func_preprocesing_audio_to_spectrogram(var_path_one_file_str = var_path_train_not_normal_breath_wav, test_var = test_var)
  '''

  file_contents = tf.io.read_file(var_path_one_file_str)
  wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
  wav = tf.squeeze(wav, axis=-1)
  sample_rate = tf.cast(sample_rate, dtype=tf.int64)
  wav_16000hz = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)

  wav_1 = wav_16000hz[:test_var]
  zero_padding_1 = tf.zeros([test_var] - tf.shape(wav_1), dtype=tf.float32)
  wav_1 = tf.concat([zero_padding_1, wav_1],0)
  spectrogram_1 = tf.signal.stft(wav_1, frame_length=320, frame_step=32)
  spectrogram_1 = tf.abs(spectrogram_1)
  spectrogram_1 = tf.expand_dims(spectrogram_1, axis=2)

  return spectrogram_1

test_var = 32000
spectr_1 = func_preprocesing_audio_to_spectrogram(var_path_one_file_str = var_path_train_normal_breath_wav, test_var = test_var)
spectr_2 = func_preprocesing_audio_to_spectrogram(var_path_one_file_str = var_path_train_not_normal_breath_wav, test_var = test_var)


class PrepareDataAudioFile:
  def __init__(self, patch_input_folder, patch_output_folder):
    self.patch_input_folder = patch_input_folder  # папка с файлами которые надо обработать
    self.patch_output_folder = patch_output_folder  # папка с файлами которые надо обработать

  def prepare_audio_file_tensor_flow(self):
    # заходим в основную папку и создаем список подпапок
    # Use os.listdir() to get a list of all files and directories in the folder
    var_name_folders_list = os.listdir(self.pach_input_folder)

    for var_name_folder_str in var_name_folders_list:

      var_path_to_under_folder_str = str(self.pach_input_folder) + '/' + str(var_name_folder_str)
      var_path_to_under_folder_str = str(var_path_to_under_folder_str)

      var_name_file_list = os.listdir(var_path_to_under_folder_str)

      # оставим в списке файлов только те которые заканчиваются на mp3
      var_name_mp3_file_list = [file for file in var_name_file_list if file.endswith('.mp3')]
      # оставим в списке файлов только те которые заканчиваются на mp3
      var_name_wav_file_list = [file for file in var_name_file_list if file.endswith('.wav')]

      # ДЛЯ СПИСКА С wav
      for var_name_file_str in var_name_wav_file_list:
        var_path_to_file_str = str(self.pach_input_folder) + '/' + str(var_name_folder_str) + '/' + str(
          var_name_file_str)
        var_path_to_output_order_and_file_str = str(self.pach_output_folder) + '/' + str(
          var_name_folder_str) + '/' + str(var_name_file_str)

        var_path_to_file_str = str(var_path_to_file_str)
        var_path_to_output_order_and_file_str = str(var_path_to_output_order_and_file_str)

        var_object_audio = convert_to_16000_hrz(file_for_processing_to_16000_hrz=var_path_to_file_str)

        var_object_for_writh_to_google_drive = SaveResults(name_object_for_save=var_object_audio,
                                                           patch_output_folder=var_path_to_output_order_and_file_str)

        # ДЛЯ СПИСКА С mp3
      for var_name_file_str in var_name_mp3_file_list:
        var_path_to_file_str = str(self.pach_input_folder) + '/' + str(var_name_folder_str) + '/' + str(
          var_name_file_str)
        var_path_to_output_order_and_file_str = str(self.pach_output_folder) + '/' + str(
          var_name_folder_str) + '/' + str(var_name_file_str)

        var_path_to_file_str = str(var_path_to_file_str)
        var_path_to_output_order_and_file_str = str(var_path_to_output_order_and_file_str)

        # del .mp3 from name file
        var_name_file_without_mp3_or_wav = var_name_file_str[:-4]

        var_convert_wav_file = convert_mp3_to_wav(var_pach_to_file=var_path_to_file_str,
                                                  name_file_without_mp3_or_wav=var_name_file_without_mp3_or_wav)

        var_object_audio = convert_to_16000_hrz(file_for_processing_to_16000_hrz=var_convert_wav_file)

        var_object_for_writh_to_google_drive = SaveResults(name_object_for_save=var_object_audio,
                                                           patch_output_folder=var_path_to_output_order_and_file_str)

  def convert_to_16000_hrz(file_for_processing_to_16000_hrz):
    '''
    !pip install tensorflow_io

    from pydub import AudioSegment
    import os
    from IPython import display
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import tensorflow_hub as hub
    import tensorflow_io as tfio
    '''

    file_contents = tf.io.read_file(file_for_processing_to_16000_hrz)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

  def convert_mp3_to_wav(var_pach_to_file, name_file_without_mp3_or_wav):
    """
    !pip install pydub ffmpeg-python
    from pydub import AudioSegment
    """
    # Convert it to WAV format
    audio = AudioSegment.from_mp3(var_pach_to_file)
    wav_file = name_file_without_mp3_or_wav + ".wav"
    audio.export(wav_file, format="wav")
    return wav_file




















