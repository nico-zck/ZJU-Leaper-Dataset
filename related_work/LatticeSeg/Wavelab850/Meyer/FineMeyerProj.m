function fdpjf = FineMeyerProj(alpha,j,n,deg)
% FineMeyerProj -- Invert Meyer Transform, finest level J-1
%  Usage
%    dpjf = FineMeyerProj(alpha,j,n,deg)
%  Inputs
%    alpha    Meyer Wavelet Coefficients, dyadic length 2^j
%    j        resolution level, 2^j = n/2
%    n        length of signal vector (must be of the form 2^J)
%    deg      degree of Meyer window (e.g. 3)
%  Outputs	
%    dpjf     projection of signal onto approximation space V_J-1
%             (length(cpjf) = n)
%
%  See Also
%    UnfoldMeyer, dct_ii, dst_ii, SeparateCoeff, ExtendProj
%

 
% *** Set end points and separate signal into real and imag parts.

	lendp = 2^(j-1); rendp = 2^j;
  
% *** Calculate trigonometric coefs from wavelet coefficients.

	[rtrigcoefs, itrigcoefs] = SeparateCoeff(alpha,'t');

% *** Calculate projection of real part of \hat f (even)
% Take DST-II of local sine coefficients.

	rtrigrec = dst_ii(rtrigcoefs);

% Unfold trigonometric reconstruction w/ (-,+) polarity.

	unflde = UnfoldMeyer(rtrigrec,[lendp,rendp],'mp','t',deg);

% Extend unfolded signal to integers -n/2+1 -> n/2.

	eextproj = ExtendProj(unflde,n,'t',[lendp,rendp],'e');

% *** Calculate projection of imaginary part of \hat f (odd)
% Take DCT-II of local cosine coefficients.

	itrigrec = dct_ii(itrigcoefs);

% Unfold trigonometric reconstruction w/ (+,-) polarity.

	unfldo = UnfoldMeyer(itrigrec,[lendp,rendp],'pm','t',deg);

% Extend unfolded signal to integers -n/2+1 -> n/2 .

	oextproj = ExtendProj(unfldo,n,'t',[lendp,rendp],'o');

%*** Combine real and imaginary parts to yield coarse level
%*** projection of \hat f.

	fdpjf = ( eextproj + i .* oextproj ) ;

%
%  Prepared for the thesis of Eric Kolaczyk, Stanford University, 1994
%  Copyright (c) Eric Kolaczyk, 1994.
%  

 
 
%
%  Part of Wavelab Version 850
%  Built Tue Jan  3 13:20:40 EST 2006
%  This is Copyrighted Material
%  For Copying permissions see COPYING.m
%  Comments? e-mail wavelab@stat.stanford.edu 
