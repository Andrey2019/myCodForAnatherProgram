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