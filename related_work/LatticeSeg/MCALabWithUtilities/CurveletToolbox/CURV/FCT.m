function C = FCT(x, is_real, nbscales, nbangles_coarse)

% FCT.m - Fast Curvelet Transform
%
% Inputs
%   x           M-by-N matrix
%
% Optional Inputs
%   is_real     forces a real-valued or complex-valued transform.
%                   0: complex-valued curvelets
%                   1: real-valued curvelets
%                   2: real-valued Wilson curvelets
%               [default set to 1 if x is real-valued, 0 otherwise]
%   nbscales    number of scales including the coarsest wavelet level
%               [default set to ceil(log2(min(M,N))/2)]
%   nbangles_coarse
%               number of angles at the 2nd coarsest level, minimum 8,
%               must be a multiple of 4. [default set to 16]
%
% Outputs
%   C           Cell array of curvelet coefficients: C{j}{l}(k1,k2) is
%               the coefficient at
%                   - scale j: integer, from finest to coarsest scale,
%                   - angle l: integer, starts at the top-left corner and
%                   increases clockwise,
%                   - position k1,k2: both integers, size varies with j
%                   and l.
%               C{end} is a vector containing additional information:
%                   size(x,1), size(x,2), is_real
%
% See also FICT.m
%
% Copyright (c) Laurent Demanet, 2004

X = fftshift(fft2(x))/sqrt(prod(size(x)));
[N1,N2] = size(X);
if nargin < 2, is_real = prod(prod(isreal(x))); end;
if nargin < 3, nbscales = ceil(log2(min(N1,N2))/2); end;
if nargin < 4, nbangles_coarse = 16; end;

% Initialization: data structure
nbangles = [nbangles_coarse .* 2.^(ceil((nbscales-(2:nbscales))/2)), 1];
        % number of angles as a function of scale
C = cell(1,nbscales+1);
for j = 1:nbscales
    C{j} = cell(1,nbangles(j));
end;

% Initialization: smooth periodic extension of high frequencies
M1 = N1/3;
M2 = N2/3;
bigN1 = 2*floor(2*M1)+1;
bigN2 = 2*floor(2*M2)+1;
equiv_index_1 = 1+mod(floor(N1/2)-floor(2*M1)+(1:bigN1)-1,N1);
equiv_index_2 = 1+mod(floor(N2/2)-floor(2*M2)+(1:bigN2)-1,N2);
X = X(equiv_index_1,equiv_index_2);
        % Invariant: equiv_index_1(floor(2*M1)+1) == (N1 + 2 - mod(N1,2))/2
        % is the center in frequency. Same for M2, N2.
window_length_1 = floor(2*M1) - floor(M1) - 1 - (mod(N1,3)==0);
window_length_2 = floor(2*M2) - floor(M2) - 1 - (mod(N2,3)==0);
        % Invariant: floor(M1) + floor(2*M1) == N1 - (mod(M1,3)~=0)
        % Same for M2, N2.
coord_1 = 0:(1/window_length_1):1;
coord_2 = 0:(1/window_length_2):1;
[wl_1,wr_1] = wedgewindow(coord_1);
[wl_2,wr_2] = wedgewindow(coord_2);
lowpass_1 = [wl_1, ones(1,2*floor(M1)+1), wr_1];
if mod(N1,3)==0, lowpass_1 = [0, lowpass_1, 0]; end;
lowpass_2 = [wl_2, ones(1,2*floor(M2)+1), wr_2];
if mod(N2,3)==0, lowpass_2 = [0, lowpass_2, 0]; end;
lowpass = lowpass_1'*lowpass_2;
Xlow = X .* lowpass;

% Loop: pyramidal scale decomposition
for j = 1:(nbscales-1),

    M1 = M1/2;
    M2 = M2/2;
    window_length_1 = floor(2*M1) - floor(M1) - 1;
    window_length_2 = floor(2*M2) - floor(M2) - 1;
    coord_1 = 0:(1/window_length_1):1;
    coord_2 = 0:(1/window_length_2):1;
    [wl_1,wr_1] = wedgewindow(coord_1);
    [wl_2,wr_2] = wedgewindow(coord_2);
    lowpass_1 = [wl_1, ones(1,2*floor(M1)+1), wr_1];
    lowpass_2 = [wl_2, ones(1,2*floor(M2)+1), wr_2];
    lowpass = lowpass_1'*lowpass_2;
    hipass = sqrt(1 - lowpass.^2);
    Xhi = Xlow;                 % size is 2*floor(4*M1)+1 - by - 2*floor(4*M2)+1
    Xlow_index_1 = ((-floor(2*M1)):floor(2*M1)) + floor(4*M1) + 1;
    Xlow_index_2 = ((-floor(2*M2)):floor(2*M2)) + floor(4*M2) + 1;
    Xlow = Xlow(Xlow_index_1, Xlow_index_2);
    Xhi(Xlow_index_1, Xlow_index_2) = Xlow .* hipass;
    Xlow = Xlow .* lowpass;     % size is 2*floor(2*M1)+1 - by - 2*floor(2*M2)+1
    
    % Loop: angular decomposition
    l = 0;
    nbquadrants = 2 + 2*(~is_real);
    nbangles_perquad = nbangles(j)/nbquadrants;
    for quadrant = 1:nbquadrants

        M_horiz = M2 * (mod(quadrant,2)==1) + M1 * (mod(quadrant,2)==0);
        M_vert = M1 * (mod(quadrant,2)==1) + M2 * (mod(quadrant,2)==0);
        if mod(nbangles_perquad,2),
            wedge_ticks_left = round((0:(1/(2*nbangles_perquad)):.5)*2*floor(4*M_horiz) + 1);
            wedge_ticks_right = 2*floor(4*M_horiz) + 2 - wedge_ticks_left;
            wedge_ticks = [wedge_ticks_left, wedge_ticks_right(end:-1:1)];
        else
            wedge_ticks_left = round((0:(1/(2*nbangles_perquad)):.5)*2*floor(4*M_horiz) + 1);
            wedge_ticks_right = 2*floor(4*M_horiz) + 2 - wedge_ticks_left;
            wedge_ticks = [wedge_ticks_left, wedge_ticks_right((end-1):-1:1)];
        end;
        wedge_endpoints = wedge_ticks(2:2:(end-1));         % integers
        wedge_midpoints = (wedge_endpoints(1:(end-1)) + wedge_endpoints(2:end))/2;
                % integers or half-integers
        
        % Left corner wedge
        l = l+1;
        first_wedge_endpoint_vert = round(2*floor(4*M_vert)/(2*nbangles_perquad) + 1);
        length_corner_wedge = floor(4*M_vert) - floor(M_vert) + ceil(first_wedge_endpoint_vert/4);
        Y_corner = 1:length_corner_wedge;
        XX = meshgrid(1:(2*floor(4*M_horiz)+1),Y_corner);
        width_wedge = wedge_endpoints(2) + wedge_endpoints(1) - 1;
        slope_wedge = (floor(4*M_horiz) + 1 - wedge_endpoints(1))/floor(4*M_vert);
        left_line = round(2 - wedge_endpoints(1) + slope_wedge*(Y_corner - 1));
                                                            % integers
        [wrapped_data, wrapped_XX] = deal(zeros(length_corner_wedge,width_wedge));
        for row = Y_corner
            cols = left_line(row) + mod((0:(width_wedge-1))-(left_line(row)-left_line(1)),width_wedge);
            admissible_cols = round(1/2*(cols+1+abs(cols-1)));
            wrapped_data(row,:) = Xhi(row,admissible_cols) .* (cols > 0);
            wrapped_XX(row,:) = XX(row,admissible_cols);
        end;
        YY = Y_corner'*ones(1,width_wedge);
        slope_wedge_right = (floor(4*M_horiz)+1 - wedge_midpoints(1))/floor(4*M_vert);
        mid_line_right = wedge_midpoints(1) + slope_wedge_right*(YY - 1);
                % not integers in general
        coord_right = 1/2 + floor(4*M_vert)/(wedge_endpoints(2) - wedge_endpoints(1)) * ...
            (wrapped_XX - mid_line_right)./(floor(4*M_vert)+1 - YY);
        C2 = 1/(1/(2*(floor(4*M_horiz))/(wedge_endpoints(1) - 1) - 1) + 1/(2*(floor(4*M_vert))/(first_wedge_endpoint_vert - 1) - 1));
        C1 = C2 / (2*(floor(4*M_vert))/(first_wedge_endpoint_vert - 1) - 1);
        wrapped_XX((wrapped_XX - 1)/floor(4*M_horiz) + (YY-1)/floor(4*M_vert) == 2) = ...
            wrapped_XX((wrapped_XX - 1)/floor(4*M_horiz) + (YY-1)/floor(4*M_vert) == 2) + 1;
        coord_corner = C1 + C2 * ((wrapped_XX - 1)/(floor(4*M_horiz)) - (YY - 1)/(floor(4*M_vert))) ./ ...
            (2-((wrapped_XX - 1)/(floor(4*M_horiz)) + (YY - 1)/(floor(4*M_vert))));
        wl_left = wedgewindow(coord_corner);
        [wl_right,wr_right] = wedgewindow(coord_right);
        wrapped_data = wrapped_data .* wl_left .* wr_right;

        switch is_real
            case 0
                wrapped_data = rot90(wrapped_data,-(quadrant-1));
                C{j}{l} = ifft2(wrapped_data)*sqrt(prod(size(wrapped_data)));
            case 1
                padded_data = [wrapped_data; zeros(size(wrapped_data,1)+1,size(wrapped_data,2))];
                padded_data = rot90(padded_data,-(quadrant-1));
                x = ifft2(padded_data)*sqrt(prod(size(padded_data)));
                C{j}{l} = 2*real(x);
            case 2
                wrapped_data = rot90(wrapped_data,-(quadrant-1));
                x = ifft2(wrapped_data)*sqrt(prod(size(wrapped_data)));
                C{j}{l} = zeros(2,size(x,1),size(x,2));
                C{j}{l}(1,:,:) = sqrt(2)*real(x);
                C{j}{l}(2,:,:) = sqrt(2)*imag(x);
        end;
                
        % Regular wedges
        length_wedge = floor(4*M_vert) - floor(M_vert);
        Y = 1:length_wedge;
        for subl = 2:(nbangles_perquad-1);
            l = l+1;
            width_wedge = wedge_endpoints(subl+1) - wedge_endpoints(subl-1) + 1;
            slope_wedge = ((floor(4*M_horiz)+1) - wedge_endpoints(subl))/floor(4*M_vert);
            left_line = round(wedge_endpoints(subl-1) + slope_wedge*(Y - 1));
            [wrapped_data, wrapped_XX] = deal(zeros(length_wedge,width_wedge));
            for row = Y
                cols = left_line(row) + mod((0:(width_wedge-1))-(left_line(row)-left_line(1)),width_wedge);
                wrapped_data(row,:) = Xhi(row,cols);
                wrapped_XX(row,:) = XX(row,cols);
            end;
            YY = Y'*ones(1,width_wedge);
            slope_wedge_left = ((floor(4*M_horiz)+1) - wedge_midpoints(subl-1))/floor(4*M_vert);
            mid_line_left = wedge_midpoints(subl-1) + slope_wedge_left*(YY - 1);
            coord_left = 1/2 + floor(4*M_vert)/(wedge_endpoints(subl) - wedge_endpoints(subl-1)) * ...
                (wrapped_XX - mid_line_left)./(floor(4*M_vert)+1 - YY);
            slope_wedge_right = ((floor(4*M_horiz)+1) - wedge_midpoints(subl))/floor(4*M_vert);
            mid_line_right = wedge_midpoints(subl) + slope_wedge_right*(YY - 1);
            coord_right = 1/2 + floor(4*M_vert)/(wedge_endpoints(subl+1) - wedge_endpoints(subl)) * ...
                (wrapped_XX - mid_line_right)./(floor(4*M_vert)+1 - YY);
            wl_left = wedgewindow(coord_left);
            [wl_right,wr_right] = wedgewindow(coord_right);
            wrapped_data = wrapped_data .* wl_left .* wr_right;
            switch is_real
                case 0
                    wrapped_data = rot90(wrapped_data,-(quadrant-1));
                    C{j}{l} = ifft2(wrapped_data)*sqrt(prod(size(wrapped_data)));
                case 1
                    padded_data = [wrapped_data; zeros(size(wrapped_data,1)+1,size(wrapped_data,2))];
                    padded_data = rot90(padded_data,-(quadrant-1));
                    x = ifft2(padded_data)*sqrt(prod(size(padded_data)));
                    C{j}{l} = 2*real(x);
                case 2
                    wrapped_data = rot90(wrapped_data,-(quadrant-1));
                    x = ifft2(wrapped_data)*sqrt(prod(size(wrapped_data)));
                    C{j}{l} = zeros(2,size(x,1),size(x,2));
                    C{j}{l}(1,:,:) = sqrt(2)*real(x);
                    C{j}{l}(2,:,:) = sqrt(2)*imag(x);
            end;
        end;

        % Right corner wedge
        l = l+1;
        width_wedge = 4*floor(4*M_horiz) + 3 - wedge_endpoints(end) - wedge_endpoints(end-1);
        slope_wedge = ((floor(4*M_horiz)+1) - wedge_endpoints(end))/floor(4*M_vert);
        left_line = round(wedge_endpoints(end-1) + slope_wedge*(Y_corner - 1));
        [wrapped_data, wrapped_XX] = deal(zeros(length_corner_wedge,width_wedge));
        for row = Y_corner
            cols = left_line(row) + mod((0:(width_wedge-1))-(left_line(row)-left_line(1)),width_wedge);
            admissible_cols = round(1/2*(cols+2*floor(4*M_horiz)+1-abs(cols-(2*floor(4*M_horiz)+1))));
            wrapped_data(row,:) = Xhi(row,admissible_cols) .* (cols <= (2*floor(4*M_horiz)+1));
            wrapped_XX(row,:) = XX(row,admissible_cols);
        end;
        YY = Y_corner'*ones(1,width_wedge);
        slope_wedge_left = ((floor(4*M_horiz)+1) - wedge_midpoints(end))/floor(4*M_vert);
        mid_line_left = wedge_midpoints(end) + slope_wedge_left*(YY - 1);
        coord_left = 1/2 + floor(4*M_vert)/(wedge_endpoints(end) - wedge_endpoints(end-1)) * ...
            (wrapped_XX - mid_line_left)./(floor(4*M_vert)+1 - YY);
        C2 = -1/(2*(floor(4*M_horiz))/(wedge_endpoints(end) - 1) - 1 + 1/(2*(floor(4*M_vert))/(first_wedge_endpoint_vert - 1) - 1));
        C1 = -C2 * (2*(floor(4*M_horiz))/(wedge_endpoints(end) - 1) - 1);
        wrapped_XX((wrapped_XX - 1)/floor(4*M_horiz) == (YY-1)/floor(4*M_vert)) = ...
            wrapped_XX((wrapped_XX - 1)/floor(4*M_horiz) == (YY-1)/floor(4*M_vert)) - 1;
        coord_corner = C1 + C2 * (2-((wrapped_XX - 1)/(floor(4*M_horiz)) + (YY - 1)/(floor(4*M_vert)))) ./ ...
            ((wrapped_XX - 1)/(floor(4*M_horiz)) - (YY - 1)/(floor(4*M_vert)));
        wl_left = wedgewindow(coord_left);
        [wl_right,wr_right] = wedgewindow(coord_corner);

        wrapped_data = wrapped_data .* wl_left .* wr_right;
        switch is_real
            case 0
                wrapped_data = rot90(wrapped_data,-(quadrant-1));
                C{j}{l} = ifft2(wrapped_data)*sqrt(prod(size(wrapped_data)));
            case 1
                padded_data = [wrapped_data; zeros(size(wrapped_data,1)+1,size(wrapped_data,2))];
                padded_data = rot90(padded_data,-(quadrant-1));
                x = ifft2(padded_data)*sqrt(prod(size(padded_data)));
                C{j}{l} = 2*real(x);
            case 2
                wrapped_data = rot90(wrapped_data,-(quadrant-1));
                x = ifft2(wrapped_data)*sqrt(prod(size(wrapped_data)));
                C{j}{l} = zeros(2,size(x,1),size(x,2));
                C{j}{l}(1,:,:) = sqrt(2)*real(x);
                C{j}{l}(2,:,:) = sqrt(2)*imag(x);
        end;

        if quadrant < nbquadrants, Xhi = rot90(Xhi); end;
    end;
end;

% Coarsest wavelet level
C{nbscales}{1} = ifft2(ifftshift(Xlow))*sqrt(prod(size(Xlow)));
if ~~is_real, C{nbscales}{1} = real(C{nbscales}{1}); end;

% Additional information
C{end} = [N1, N2, is_real];
