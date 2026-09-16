% Referencni vystup filterGaussianFit2D.m (cisty MATLAB, bez MEXu)
% na realnem snimku -- pro validaci portu cmepython.background.
ROOT = fileparts(fileparts(mfilename('fullpath')));      % koren repozitare
addpath(genpath(fullfile(ROOT, 'cmeAnalysis', 'software')));
TIF = fullfile(ROOT, 'reconstructed registered', ...
               'U2OS_DYNAMIN_MSTAYGOLD_GREEN_SNAP_CLC_RED_DNMsiRNA_16_RR.tif');
SIGMA = 2.6424; NCH = 3; FRAME = 51; CH = 3;      % 1-based
img = double(imread(TIF, (FRAME-1)*NCH + CH));
[A_est, c_est, pval] = filterGaussianFit2D(img, SIGMA);
save(fullfile(ROOT, 'matlab', 'filter_ref.mat'), ...
     'A_est','c_est','pval','SIGMA','-v7');
fprintf('exportovano %dx%d, A median %.2f\nHOTOVO\n', size(A_est), median(A_est(:)));
