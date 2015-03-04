# author: mshadish
################################################################################
# This script contains some of the EDA performed
# on the features extracted from the driver telematics data
################################################################################

# WE WILL DEFINE A FUNCTION THAT CAN TAKE IN A VECTOR OF DATA FRAME NUMBERS
# AND CREATE HISTOGRAMS OF THE SPECIFIED COLUMN
# FOR EACH DATA FRAME
genPlots = function(list_numbers, col_name,
                    directory = '.', suffix = '_summary.csv') {
  # first, clear any existing plots that may be present
  dev.off()
  # step through the numbers in the list passed in
  for (num in list_numbers) {
    # generate a file name
    file_name = paste(num, suffix, sep = '')
    # and test to verify it is in the given directory
    if (!file_name %in% list.files(directory)) {
      print(paste('File', num,'not found', sep = ' '))
      next
    }
    # otherwise, we are free to open the file
    data = read.csv(paste(directory, file_name, sep = '/'))
    # and plot a histogram of the parameter specified
    hist(data[[col_name]], main = paste(num, col_name, sep = ' '))
  }
}

##########################################
# let's generate a random sample of 10 files
rand_files = sample(1:3612, size = 15)
# note: the file naming scheme is strange,
# so it's normal to not find some files between 1 and 3612
# the function will take care of this
##########################################
# LEFT TURN FRACTION

##########################################
# AVG LEFT TURN ANGLE

##########################################
# MEDIAN LEFT TURN ANGLE

##########################################
# MAXIMUM LEFT TURN ANGLE

##########################################
# SD OF LEFT TURN ANGLES

##########################################
# RIGHT TURN FRACTION

##########################################
# AVERAGE RIGHT TURN ANGLE

##########################################
# MEDIAN RIGHT TURN ANGLE

##########################################
# MAXIMUM RIGHT TURN ANGLE

##########################################
# SD OF RIGHT TURN ANGLES

##########################################
# FINAL DIRECTION

##########################################
# AVERAGE VELOCITY

##########################################
# MEDIAN VELOCITY

##########################################
# AVERAGE VELOCITY, WITH 0'S REMOVED

##########################################
# MEDIAN VELOCITY, 0'S REMOVED

##########################################
# MAXIMUM VELOCITY ACHIEVED

##########################################
# TIME SPENT CRUISING

##########################################
# TOTAL DISTANCE TRAVELED
genPlots(rand_files, 'distance_l')
# distance traveled shows some decent variation in the distributions

##########################################
# MAXIMUM ACCELERATION ACHIEVED

##########################################
# MAXIMUM BRAKING

##########################################
# TIME SPENT ACCELERATING

##########################################
# TIME SPENT BRAKING

##########################################
# AVERAGE ACCELERATION
genPlots(rand_files, 'avg_accel_l')
# average acceleration appears heavily skewed regardless

##########################################
# MEDIAN ACCELERATION
genPlots(rand_files, 'med_accel_l')
# med acceleration shows mild variation in distributions

##########################################
# AVERAGE DECELERATION/BRAKING

##########################################
# MEDIAN DECELERATION/BRAKING
genPlots(rand_files, 'med_braking_l')
# median braking appears to demonstrate different distributions

##########################################
