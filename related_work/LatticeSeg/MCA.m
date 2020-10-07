function [cartoon, texture] = MCA(img)

% Dictionary stuff (here Curvelets + UDWT).
dict1 = 'CURVWRAP'; 
pars11 = '2'; 
pars12 = 0; 
pars13 = 0;
% dict1 = 'CURV'; 
% pars11 = '2'; 
% pars12 = 0; 
% pars13 = 0;

dict2 = 'LDCT2iv'; 
pars21 = 'Sine'; 
pars22 = 32; % wisnow size
pars23 = 128/512; % Remove Low-frequencies 128/512 from textured part.
% dict2 = 'LDCT2'; 
% pars21 = '5'; % wisnow size, need to be str type to match with pars11
% pars22 = 128/512; 
% pars23 = 0; % Remove Low-frequencies 128/512 from textured part.


dicts = MakeList(dict1, dict2);
pars1 = MakeList(pars11, pars21);
pars2 = MakeList(pars12, pars22);
pars3 = MakeList(pars13, pars23);

% Call the MCA.
itermax = 30;
tvregparam = 2;
tvcomponent = 1;
expdecrease = 1;
lambdastop = 1;
% sigma = 1E-6;
sigma = [];
display = 0; %[0-none, 1-bar, 2-bar and figure]

[parts, options] = MCA2_Bcr(img, dicts, pars1, pars2, pars3, itermax, tvregparam, tvcomponent, expdecrease, lambdastop, [], sigma, display);
options.inputdata = 'fabric image';

cartoon = squeeze(parts(:, :, 1));
texture = squeeze(parts(:, :, 2));

end
