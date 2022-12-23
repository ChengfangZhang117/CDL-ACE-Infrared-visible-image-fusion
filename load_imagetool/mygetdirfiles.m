function files = mygetdirfiles(dirpath)
dirFile=dir(dirpath);% 
imageCount = 0;
for i = 1 : size(dirFile,1)
   if ~dirFile(i).isdir
      imageCount = imageCount + 1;
      files{imageCount,1}= fullfile(dirpath, dirFile(i).name);
   end
end