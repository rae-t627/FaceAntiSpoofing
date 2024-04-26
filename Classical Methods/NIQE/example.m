
 load modelparameters.mat
 
 blocksizerow    = 96;
 blocksizecol    = 96;
 blockrowoverlap = 0;
 blockcoloverlap = 0;

im =imread("C:\Users\Kartik\Downloads\Grayscale\Grayscale\NUAA\Detectedface\ImposterFace\0001\0001_00_00_01_0.jpg");

quality = computequality(im,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
    mu_prisparam,cov_prisparam)

% im =imread('image2.bmp');
% 
% quality = computequality(im,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
%     mu_prisparam,cov_prisparam)
% 
% im =imread('image3.bmp');
% 
% quality = computequality(im,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
%     mu_prisparam,cov_prisparam)
% 
% 
% im =imread('image4.bmp');
% 
% quality = computequality(im,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
%     mu_prisparam,cov_prisparam)