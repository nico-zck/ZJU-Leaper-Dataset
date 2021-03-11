function do_scalegram
% do_scalegram -- Called by WLBrowser
%  Usage
%    do_scalegram
%  Inputs
%    none
%  Outputs
%    none
%
%  Description
%    The scalegram is the wavelet analog of the power spectrum.
%    It is just the (squared) wavelet coefficient averaged over
%    all location index values at a fixed value of the scale index.
%
%  Side Effects
%    Scalegram is computed and plotted.
%
m=2;st='Scalegram';setfigure;
[ ndum, L ] = dyadlength( QMF_Filter );
[   nx, J ] = dyadlength( x_work );

L_use = L - 1;
  if L_use < 1
    L_use = 1;
  end

wc = FWT_PO( x_work, L_use, QMF_Filter );

for i = L:J-1
  sg(i-L+1) = mean( ( wc(dyad(i)) ).^2);
end
nsp = length( x_work )/2;
sp = wl_spectrum( x_work, nsp );
sg = reverse( sg );
sp = reverse( sp )/sqrt(nx);
tsg = (L:J-1)-1;
tsp = 1:nsp;tsp = log(tsp) ./ log(2);

plot( ...
    tsp, log10(sp),'-r', ...
    tsg, log10(sg),'-g', ...
    tsg, log10(sg),'xb' )
title( signaltitle)
xlabel('log_2(time scale)')
ylabel('log10(Power)')
    
    
 
 
%
%  Part of Wavelab Version 850
%  Built Tue Jan  3 13:20:39 EST 2006
%  This is Copyrighted Material
%  For Copying permissions see COPYING.m
%  Comments? e-mail wavelab@stat.stanford.edu 
