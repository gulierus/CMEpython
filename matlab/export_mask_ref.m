% Referencni maska bunky: maskFromFirstMode + imfill na max-projekci,
% kterou vyexportoval Python (stejny vstup pro obe implementace).
ROOT = fileparts(fileparts(mfilename('fullpath')));      % koren repozitare
addpath(genpath(fullfile(ROOT, 'cmeAnalysis', 'software')));
proj = double(imread(fullfile(ROOT, 'matlab', 'maxproj_ch0.tif')));
[mask, T] = maskFromFirstMode(proj, 'ModeRatio', 0.8);   % getCellMask.m:37
mask = imfill(mask, 'holes');                            % getCellMask.m:116
imwrite(uint8(mask*255), fullfile(ROOT, 'matlab', 'mask_ref.tif'));
fprintf('prah T = %.3f, maska %.1f %% plochy\nHOTOVO\n', T, 100*mean(mask(:)));
