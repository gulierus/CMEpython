fns = {'imread','imdilate','imfill','bwlabel','bwconncomp','strel','padarray', ...
       'tcdf','norminv','normpdf','prctile','ksdensity','statset','gmdistribution'};
for k = 1:numel(fns)
    w = which(fns{k});
    if isempty(w)
        fprintf('CHYBI   %s\n', fns{k});
    else
        fprintf('OK      %-16s %s\n', fns{k}, w);
    end
end

fprintf('\n--- MEX fitGaussian2D ---\n');
ROOT = fileparts(fileparts(mfilename('fullpath')));      % koren repozitare
addpath(genpath(fullfile(ROOT, 'cmeAnalysis', 'software')));
[yy, xx] = ndgrid(-6:6, -6:6);
img = 100 + 500*exp(-((xx.^2 + yy.^2)/(2*1.5^2)));
try
    [prm, prmStd, C, res] = fitGaussian2D(img, [0 0 400 1.5 90], 'xyAc');
    fprintf('OK  A=%.4f  c=%.4f  x=%.4f  y=%.4f  RSS=%.3e\n', ...
            prm(3), prm(5), prm(1), prm(2), res.RSS);
catch e
    fprintf('SELHAL: %s\n', e.message);
end
fprintf('HOTOVO\n');
