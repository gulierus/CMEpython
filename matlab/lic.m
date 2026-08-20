[status, msg] = license('checkout','Image_Toolbox');
fprintf('Image_Toolbox checkout: %d  %s\n', status, msg);
[status2, msg2] = license('checkout','Statistics_Toolbox');
fprintf('Statistics_Toolbox checkout: %d  %s\n', status2, msg2);
fprintf('licence cislo: %s\n', license);
try
  fprintf('\n--- nainstalovane produkty ---\n');
  a = matlabshared.supportpkg.getInstalled();
  if isempty(a), fprintf('zadne support packages\n'); end
catch e
  fprintf('(getInstalled: %s)\n', e.message);
end
fprintf('HOTOVO\n');
