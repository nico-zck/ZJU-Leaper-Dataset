function s = cyclespin(x,k)
n = length(x);
if k > 0,
	s = [x((n+1-k):n) x(1:(n-k))];
else
	s = [x((-k+1):n) x(1:(-k)) ];
end
    
    
 
 
%
%  Part of Wavelab Version 850
%  Built Tue Jan  3 13:20:42 EST 2006
%  This is Copyrighted Material
%  For Copying permissions see COPYING.m
%  Comments? e-mail wavelab@stat.stanford.edu 
