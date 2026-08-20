% Vyexportuje okna z realnych dat + vysledky MEX fitu fitGaussian2D,
% aby se daly porovnat s Python portem. Zamerne obchazi fitGaussians2D.m,
% ktery potrebuje chybejici toolboxy -- volame MEX primo.

addpath(genpath('/Users/ruslanguliev/CMEpython/cmeAnalysis/software'));

TIF = ['/Users/ruslanguliev/CMEpython/reconstructed registered/' ...
       'U2OS_DYNAMIN_MSTAYGOLD_GREEN_SNAP_CLC_RED_DNMsiRNA_16_RR.tif'];
SIGMA = 2.6424;            % kanal 2, kalibrovano pres cely dataset
NCH   = 3;
FRAME = 51;                % 1-based -> odpovida Python frame 50
CH    = 3;                 % 1-based -> odpovida Python kanalu 2

page = (FRAME-1)*NCH + CH;
img  = double(imread(TIF, page));
[ny, nx] = size(img);
iRange = [min(img(:)) max(img(:))];

% pozice: pravidelna mrizka + nejjasnejsi pixely, aby byly zastoupeny
% jak prazdna mista, tak silny signal
w4 = ceil(4*SIGMA);
[gy, gx] = ndgrid(w4+5 : 37 : ny-w4-5, w4+5 : 41 : nx-w4-5);
pos = [gy(:) gx(:)];
[~, ord] = sort(img(:), 'descend');
[by, bx] = ind2sub([ny nx], ord(1:4000));
keep = by>w4+2 & by<=ny-w4-2 & bx>w4+2 & bx<=nx-w4-2;
bright = [by(keep) bx(keep)];
pos = [pos; bright(1:20:min(6000,size(bright,1)), :)];
n = size(pos,1);

% mezikruzi pro odhad pozadi -- fitGaussians2D.m:166-174
[xm, ym] = meshgrid(-w4:w4);
r = sqrt(xm.^2 + ym.^2);
annular = r <= ceil(4*SIGMA) & r >= ceil(3*SIGMA);

wins    = zeros(2*w4+1, 2*w4+1, n);
init    = zeros(n, 5);
res_Ac  = nan(n, 5);   res_Ac_std  = nan(n, 5);
res_xy  = nan(n, 5);   res_xy_std  = nan(n, 5);
rss_Ac  = nan(n, 1);   rss_xy  = nan(n, 1);
std_Ac  = nan(n, 1);   std_xy  = nan(n, 1);
hAD_Ac  = nan(n, 1);   pv_Ac   = nan(n, 1);

for k = 1:n
    yi = pos(k,1); xi = pos(k,2);
    win = img(yi-w4:yi+w4, xi-w4:xi+w4);
    wins(:,:,k) = win;

    c_init = mean(win(annular));              % :174
    A_init = max(win(:)) - c_init;            % :187
    init(k,:) = [0 0 A_init SIGMA c_init];

    % POZOR: prmStd ma delku jen podle poctu VOLNYCH parametru
    % ('Ac' -> 2, 'xyAc' -> 4). fitGaussians2D.m:207 je rozptyluje
    % pres stdVect(estIdx) = prmStd.
    [p1, s1, ~, r1] = fitGaussian2D(win, init(k,:), 'Ac');
    res_Ac(k,:) = p1;
    res_Ac_std(k,[3 5]) = s1;                 % regexpi('xyAsc','[Ac]') == [3 5]
    rss_Ac(k) = r1.RSS; std_Ac(k) = r1.std;
    if isfield(r1,'hAD'), hAD_Ac(k) = r1.hAD; end
    if isfield(r1,'pval'), pv_Ac(k) = r1.pval; end

    [p2, s2, ~, r2] = fitGaussian2D(win, [0 0 p1(3) SIGMA p1(5)], 'xyAc');
    res_xy(k,:) = p2;
    res_xy_std(k,[1 2 3 5]) = s2;             % regexpi('xyAsc','[xyAc]') == [1 2 3 5]
    rss_xy(k) = r2.RSS; std_xy(k) = r2.std;
end

save('/Users/ruslanguliev/CMEpython/matlab/mex_reference.mat', ...
     'wins','init','pos','res_Ac','res_Ac_std','res_xy','res_xy_std', ...
     'rss_Ac','rss_xy','std_Ac','std_xy','hAD_Ac','pv_Ac','SIGMA','iRange','w4','-v7');

fprintf('exportovano %d oken, w4=%d, sigma=%.4f\n', n, w4, SIGMA);
fprintf('A(Ac) rozsah %.1f az %.1f, median %.1f\n', ...
        min(res_Ac(:,3)), max(res_Ac(:,3)), median(res_Ac(:,3)));
fprintf('HOTOVO\n');
