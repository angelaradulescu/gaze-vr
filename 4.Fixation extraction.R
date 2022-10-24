
library(emov)

## Load data.
data_direc = '/Users/angelaradulescu/Dropbox (Facebook)/VisualSearch/VisualSearchCode/Gaze/fixation_extraction/'
gaze_path = paste(data_direc,'gaze_fix_ready.csv', sep="")
fix_path = paste(data_direc,'extracted_fixations.csv', sep="")
gaze <- read.csv(file=gaze_path, header=TRUE, sep=",")

## Set algorithm parameters.
# Maximum displacement.
max_disp = 1 
s_freq = 120
# Minimum fixation duration set to 100ms.
min_dur_ms = 88
min_dur_samples = floor(min_dur_ms * s_freq/1000)

## Run. 
start_time <- Sys.time()
fixations <- emov.idt(gaze[['timepoint']], gaze[['gaze_pos_geo_long']], gaze[['gaze_pos_geo_lat']], max_disp, min_dur_samples)
end_time <- Sys.time()

print(end_time - start_time)

## Save.
write.csv(fixations_only, fix_path)

