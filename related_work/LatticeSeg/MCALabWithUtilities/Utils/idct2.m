function x = idct2(c)
% idct2 -- 2-dimensional inverse discrete cosine transform (type I)
%  Usage
%    x = idct2(c)
%  Inputs
%    c     2-d cosine transform
%  Outputs
%    x     2-d image (n by n array, n dyadic)
%
%  Description
%    A two-dimensional inverse DCT is computed for the
%    array c. 
%
%  See Also
%    dct2, idct
%

	[nr,nc] = size(c);
	x = zeros(nr,nc);
	
	for iy=1:nc,
		col = c(:,iy)';
		x(:,iy) = idct(col)';
	end
	
	for ix=1:nr,
		row = x(ix,:);
		x(ix,:) = idct(row);
	end
	



