% Runs all 1D script files.
% Produces all the figures.

close all
clear all

%%%%%%%%%%%%
% 1D Demos.%
%%%%%%%%%%%%
disp('Starting 1D demos.');
disp('Note: running all demos.');
disp('');

disp('Decomposing synthetic Locally oscillating + Bumps');
figMCABumpsLocalSines;
disp('Decomposing synthetic Cosine + Bumps');
figMCABumpsCosine;
disp('Decomposing synthetic TwinSine + Diracs');
figMCADiracTwinSine;
disp('Decomposing Star signal');
figMCAStar;
disp('Decomposing EEG-fMRI signal');
figMCAEEG;

close(figure(1));
tilefigs




