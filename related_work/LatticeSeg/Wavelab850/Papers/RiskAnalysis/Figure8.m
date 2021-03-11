global x_use x_length x_name sigchoice  fig_hand3 PlotFig
global x_fft x_haar x_symm x_coif x_daub x_noise
global x_tmp signal_name
figure(fig_hand3);
clg;orient landscape;
x0 = log2(x_length/2);
pr_name = x_name;pr_length = x_length;
lambdMO = [1.47391  1.66865 1.85982 2.04748 2.23203 2.41405 ...
           2.59427 2.77343 2.95218 3.13093 3.30975];
% Minimax Optimal lamda derived in Donoho-Johnstone, Biometrika 94
n0 = 2.^(6:16);
for i = 1:length(n0),
    if x_length == n0(i)
      lambdMO = lambdMO(i);
    end
    if any(x_length == n0) == 0
      disp('Do not know lambda-MO for this length of signal');
    end
end
if (sum(sigchoice == 1) == 1),
   x_symm = FWT_PO(x_use,0,MakeONFilter('Symmlet',8));
   [x_symm_ideal_low,x_raw_low,x_symm_hard_low,x_symm_proj_low] = ...
      CalcThreshScaleRisk(x_symm,x_length,0.02);
   [x_symm_ideal_low,x_raw_low,x_symm_soft_low,x_symm_proj_low] = ...
      CalcThreshScaleRisk(x_symm,x_length,0.02,'Soft');
   [x_symm_ideal_low,x_raw_low,x_symm_softmo_low,x_symm_proj_low] = ...
      CalcThreshScaleRisk(x_symm,x_length,0.02,'Soft',lambdMO);
   [x_symm_ideal_high,x_raw_high,x_symm_hard_high,x_symm_proj_high] = ...
      CalcThreshScaleRisk(x_symm,x_length,0.1);
[x_symm_ideal_high,x_raw_high,x_symm_soft_high,x_symm_proj_high] = ...
      CalcThreshScaleRisk(x_symm,x_length,0.1,'Soft');
[x_symm_ideal_high,x_raw_high,x_symm_softmo_high,x_symm_proj_high] = ...
      CalcThreshScaleRisk(x_symm,x_length,0.1,'Soft',lambdMO);
      Scale = 0:log2(x_length);
ymin = min([min(x_raw_low)  ...
            min(x_symm_hard_low) min(x_symm_softmo_low) ...
            min(x_symm_soft_low) min(x_raw_high) min(x_symm_softmo_high) ...
            min(x_symm_hard_high) min(x_symm_soft_high)]); 
ymax = max([max(x_raw_low)  max(x_symm_softmo_low) ...
            max(x_symm_hard_low) max(x_symm_softmo_high) ...
            max(x_symm_soft_low) max(x_raw_high) ...
            max(x_symm_hard_high) max(x_symm_soft_high)]); 
delta = 0.05*(ymax-ymin);
   subplot(211),
      plot(Scale,x_symm_hard_low);
      hold on;v = axis;axis([v(1) v(2) (ymin-delta) (ymax+delta)]);
      plot(Scale,x_symm_soft_low,'--');
      plot(Scale,x_symm_softmo_low,'-.');
      plot(Scale,x_raw_low,':');
      xlabel('Threshold Scale');ylabel('log10(Risk)');
titlestr = ['Signal ' x_name ': sigma=0.02'];
legend('-','Hard R2Ln','--','Soft R2Ln','-.','Soft DJMO',':','Raw Y');
      title(titlestr);
      hold off;
   subplot(212),
      plot(Scale,x_symm_hard_high);
      hold on;v = axis;axis([v(1) v(2) (ymin-delta) (ymax+delta)]);
      plot(Scale,x_symm_soft_high,'--');
      plot(Scale,x_symm_softmo_high,'-.');
      plot(Scale,x_raw_high,':');
      xlabel('Threshold Scale');ylabel('log10(Risk)');
titlestr = ['Signal ' x_name ': sigma=0.1'];
      title(titlestr);
legend('-','Hard R2Ln','--','Soft R2Ln','-.','Soft DJMO',':','Raw Y');
      hold off;
else 
figure(fig_hand3);clg;orient landscape;     
      nplot = sum(sigchoice);
b = 1:10; b = b(sigchoice == 1);
  for i = 1:nplot,
    Gen_signal(b(i),'No');
    figure(fig_hand3);
   x_symm = FWT_PO(x_tmp,0,MakeONFilter('Symmlet',8));
[x_symm_ideal_low,x_raw_low,x_symm_hard_low,x_symm_proj_low] = ...
      CalcThreshScaleRisk(x_symm,x_length,0.02);
[x_symm_ideal_low,x_raw_low,x_symm_soft_low,x_symm_proj_low] = ...
      CalcThreshScaleRisk(x_symm,x_length,0.02,'Soft');
   [x_symm_ideal_low,x_raw_low,x_symm_softmo_low,x_symm_proj_low] = ...
      CalcThreshScaleRisk(x_symm,x_length,0.02,'Soft',lambdMO);
[x_symm_ideal_high,x_raw_high,x_symm_hard_high,x_symm_proj_high] = ...
      CalcThreshScaleRisk(x_symm,x_length,0.1);
[x_symm_ideal_high,x_raw_high,x_symm_soft_high,x_symm_proj_high] = ...
      CalcThreshScaleRisk(x_symm,x_length,0.1,'Soft');
[x_symm_ideal_high,x_raw_high,x_symm_softmo_high,x_symm_proj_high] = ...
      CalcThreshScaleRisk(x_symm,x_length,0.1,'Soft',lambdMO);
      Scale = 0:log2(x_length);
ymin = min([min(x_raw_low) min(x_symm_softmo_low) ...
            min(x_symm_hard_low) min(x_symm_softmo_high) ...
            min(x_symm_soft_low) min(x_raw_high) ...
            min(x_symm_hard_high) min(x_symm_soft_high)]); 
ymax = max([max(x_raw_low) max(x_symm_softmo_low) ...
            max(x_symm_hard_low)  max(x_symm_softmo_high) ...
            max(x_symm_soft_low) max(x_raw_high) ...
            max(x_symm_hard_high) max(x_symm_soft_high)]); 
delta = 0.05*(ymax-ymin);
   subplot(nplot,2,(2*(i-1)+1)),
      plot(Scale,x_symm_hard_low);
      hold on;v = axis;axis([v(1) v(2) (ymin-delta) (ymax+delta)]);
      plot(Scale,x_symm_soft_low,'--');
      plot(Scale,x_symm_softmo_low,'-.');
      plot(Scale,x_raw_low,':');
      xlabel('Threshold Scale');ylabel('log10(Risk)');
titlestr = ['Signal ' signal_name ': sigma=0.02'];
      title(titlestr);
legend('-','Hard R2Ln','--','Soft R2Ln','-.','Soft DJMO',':','Raw Y');
      hold off;
   subplot(nplot,2,(2*(i-1)+2)),
      plot(Scale,x_symm_hard_high);
      hold on;v = axis;axis([v(1) v(2) (ymin-delta) (ymax+delta)]);
      plot(Scale,x_symm_soft_high,'--');
      plot(Scale,x_symm_softmo_high,'-.');
      plot(Scale,x_raw_high,':');
titlestr = ['Signal ' signal_name ': sigma=0.02'];
      title(titlestr);
legend('-','Hard R2Ln','--','Soft R2Ln','-.','Soft DJMO',':','Raw Y');
      hold off;
end
end

displength = num2str(x_length);
disp('***************************************************************');
disp(['*      The length of the signal used =  '  displength '      *']);
disp('*                Basis used :  Symmlet 8                      *');
disp('***************************************************************');
x_name = pr_name;x_length = pr_length;
    

    
 
 
%
%  Part of Wavelab Version 850
%  Built Tue Jan  3 13:20:42 EST 2006
%  This is Copyrighted Material
%  For Copying permissions see COPYING.m
%  Comments? e-mail wavelab@stat.stanford.edu 
