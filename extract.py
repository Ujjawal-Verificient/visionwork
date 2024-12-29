import glob
import os
all_file_path=glob.glob('*.webm')
print(len(all_file_path))
cmd="ffmpeg -i"# e40fc119-18c9-4bc1-8634-101790c0b4bf_20241120224542505609_20241120224544802744_0_merged.webm -vf "fps=0.5" output/frame_%04d.jpg
for file in all_file_path:
    #print(file)
    directory=file[:-5]
    if not os.path.exists(directory):
        os.makedirs(directory)
    cmd1=cmd+" "+file+" -vf "+ '"fps=0.5"'+" "+directory+"/frame_%04d.jpg"
    print(cmd1)
    os.system(cmd1)
    #break