% Runs all script files.
% Produces all the figures.

close all
clear all

%%%%%%%%%%%%
% 2D Demos.%
%%%%%%%%%%%%
disp('Starting 2D demos.');
disp('Note: running all demos may take a while.');
disp('');

disp('Decomposing synthetic Oscillatory Texture + Gaussians');
figMCAtexturegaussians;
disp('Decomposing synthetic Piece-wise smooth + Oscillatory Texture');
figMCAdometextures;
disp('Decomposing synthetic Lines + Gaussians');
figMCAlinesgaussians;
disp('Decomposing synthetic Cartoon + Texture');
figMCAboytexture;
disp('Decomposing Barbara Cartoon + Texture');
figMCAbarbara;
disp('Decomposing Risers Cartoon + Lines');
figMCArisers;

disp('Inpainting synthetic Oscillatory Texture + Gaussians');
figInpainttexturegaussians;
disp('Inpainting Lines + Oscillatory Texture');
figInpaintlinesgaussians;
disp('Inpainting Lena with 80%% missing data and large gaps');
figInpaintlena;
disp('Inpainting Barbara with 20%%, 50%% and 80%% missing data');
figInpaintbarbara;

close(figure(1));
tilefigs

