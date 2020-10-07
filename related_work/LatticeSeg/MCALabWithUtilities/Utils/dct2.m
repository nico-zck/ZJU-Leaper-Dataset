function c = dct2(x)
% dct2 -- 2-dimensional discrete cosine transform (type I)
%  Usage
%    c = dct2(x)
%  Inputs
%    x     2-d image (n by n array, n dyadic)
%  Outputs
%    c     2-d cosine transform
%
%  Description
%    A two-dimensional DCT is computed for the
%    array x. To reconstruct, use the function:
%          x = idct2(c)
%
%  See Also
%    dct
%

	[nr,nc] = size(x);
	c = zeros(nr,nc);
	for ix=1:nr,
		row = x(ix,:);
		c(ix,:) = dct(row);
	end
	for iy=1:nc,
		col = c(:,iy)';
		c(:,iy) = dct(col)';
	end



