clc;

real_folder = 'C:\Users\Kartik\OneDrive - IIT Hyderabad\Desktop\Acad_IITH(6th Sem)\IVP\Dataset\replay_attack_cropped\Dataset\Replay Attack\Dataset\test\attack\hand\';

subFolders = dir(real_folder);
subFolders = subFolders([subFolders.isdir]);
subFolders = subFolders(~ismember({subFolders.name}, {'.', '..'})); % Remove '.' and '..'
niqe_arr = [];

% Loop through each subfolder
for i = 1:numel(subFolders)

    currentFolder = fullfile(real_folder, subFolders(i).name);
    real_files = dir(fullfile(currentFolder, '*.jpg'));
    num_images = length(real_files);
    niqe_values = zeros(1, num_images);

    for j = 1:num_images
        % Read real and spoof images
        imdist = imread(fullfile(currentFolder, real_files(j).name));
        imdist_resize = imresize(imdist, [64 64]);
        imdist_resize = im2double(imdist_resize);
        niqe_values(j) = niqe(imdist_resize);
    end

    niqe_values
    niqe_arr = [niqe_arr, niqe_values];

end

writematrix(niqe_arr, 'niqe_test_attack_hand.csv');

% I = imread('image1.bmp');
% I = im2double(I);
% 
% niqeI = niqe(I);
% fprintf('NIQE score for original image is %0.4f.\n',niqeI)