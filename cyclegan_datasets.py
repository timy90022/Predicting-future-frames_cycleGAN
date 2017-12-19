class cyclegan_datasets(object):  
  """Contains the standard train/test splits for the cyclegan data."""
  
  """The size of each dataset. Usually it is the maximum number of images from
  each domain."""
  DATASET_TO_SIZES = {
      'basketball_predict_train': 1200,    # 'horse2zebra_train': 1334,
      'basketball_predict_test': 140       # 'horse2zebra_test': 140
  }
  
  """The image types of each dataset. Currently only supports .jpg or .png"""
  DATASET_TO_IMAGETYPE = {
      'basketball_predict_train': '.jpg',  # 'horse2zebra_train': '.jpg',
      'basketball_predict_test': '.jpg',   # 'horse2zebra_test': '.jpg',
  }
  
  """The path to the output csv file."""
  PATH_TO_CSV = {
      'basketball_predict_train': './input/basketball_predict/basketball_predict_train.csv',  
      'basketball_predict_test': './input/basketball_predict/basketball_predict_test.csv',
      
      # 'horse2zebra_train': './input/horse2zebra/horse2zebra_train.csv',
      # 'horse2zebra_test': './input/horse2zebra/horse2zebra_test.csv',
  }
