function [Lw, Lh, Sx, Sy] = latticeSeg(cartoon)
% Lw, lattice width, on horizontal
% Lh, lattice heigh, on vertical
[m, n] = size(cartoon);

y = smoothts(mean(cartoon, 1) , 'g', 10, 5);
x = smoothts(mean(cartoon, 2)', 'g', 10, 5); %only smooth over series along row direction

% figure;
% subplot(221);plot(mean(cartoon, 1));subplot(222);plot(y);
% subplot(223);plot(mean(cartoon, 2));subplot(224);plot(x);

[yp, Sy] = findpeaks(-x(:));
[xp, Sx] = findpeaks(-y(:));
% [yp, Sy] = findpeaks(x(:));
% [xp, Sx] = findpeaks(y(:));
% Sy = Sy([~(diff(Sy)<(Lw/2));true]);
% Sx = Sx([~(diff(Sx)<(Lw/2));true]);
% Sy = Sy + Lh / 2;
% Sx = Sx + Lw / 2;
% Sy = round(Sy);
% Sx = round(Sx);

Lh = mean(diff(Sy));
Lw = mean(diff(Sx));
Lh = round(Lh);
Lw = round(Lw);

Sy = Sy(Sy < n);
Sx = Sx(Sx < m);
end