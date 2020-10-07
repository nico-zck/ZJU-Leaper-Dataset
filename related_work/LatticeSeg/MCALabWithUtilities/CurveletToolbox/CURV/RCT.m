function C = RCT(x, is_real, nbscales, nbangles_coarse)

% RCT.m - Redundant Curvelet Transform
%
% Can be seen as a numerically invertible discretization of the Continuous
% Curvelet Transform on the rectangular grid of the original image.
%
% Inputs
%   x           M-by-N matrix
%
% Optional Inputs
%   is_real     forces a real-valued or complex-valued transform.
%                   0: complex-valued curvelets
%                   1: real-valued Wilson curvelets
%               [default set to 1 if x is real-valued, 0 otherwise]
%   nbscales    number of scales including the coarsest wavelet level
%               [default set to ceil(log2(min(M,N))/2)]
%   nbangles_coarse
%               number of angles at the 2nd coarsest level, minimum 8,
%               must be a multiple of 4. [default set to 16]
%
% Outputs
%   C           Cell array of curvelet coefficients: C{j}{l}(t,k1,k2) is
%               the coefficient at
%                   - scale j: integer, from finest to coarsest scale,
%                   - angle l: integer, starts at the top-left corner and
%                   increases clockwise,
%                   - type t: 1 for 'cosine', 2 for 'sine' curvelets,
%                   - position k1, k2: both integers, k1 takes on N values
%                   and k2 M values, independently of j and l.
%               C{end} is a vector containing additional information:
%                   [size(x,1), size(x,2), is_real]
%
% See also IRCT.m
%
% Copyright (c) Laurent Demanet, 2004

X = fftshift(fft2(x))/sqrt(prod(size(x)));
[N1,N2] = size(X);
if nargin < 2, is_real = isreal(x); end;
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

% Preparation of folding steps
bigM1 = N1/3;
bigM2 = N2/3;
shift_1 = floor(2*bigM1)-floor(N1/2);
shift_2 = floor(2*bigM2)-floor(N2/2);

% Loop: pyramidal scale decomposition
Xj_topleft_1 = 1;
Xj_topleft_2 = 1;
for j = 1:(nbscales-1),

    M1 = M1/2;
    M2 = M2/2;
    loc_1 = Xj_topleft_1 + (0:(2*floor(4*M1)));
    loc_2 = Xj_topleft_2 + (0:(2*floor(4*M2)));
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
        width_wedge = 2*floor(4*M_horiz)+1;
        XX = meshgrid(1:width_wedge,Y_corner);
        data = Xhi(Y_corner,1:width_wedge);
        YY = Y_corner'*ones(1,width_wedge);
        slope_wedge_right = (floor(4*M_horiz)+1 - wedge_midpoints(1))/floor(4*M_vert);
        mid_line_right = wedge_midpoints(1) + slope_wedge_right*(YY - 1);
                % not integers in general
        coord_right = 1/2 + floor(4*M_vert)/(wedge_endpoints(2) - wedge_endpoints(1)) * ...
            (XX - mid_line_right)./(floor(4*M_vert)+1 - YY);
        C2 = 1/(1/(2*(floor(4*M_horiz))/(wedge_endpoints(1) - 1) - 1) + 1/(2*(floor(4*M_vert))/(first_wedge_endpoint_vert - 1) - 1));
        C1 = C2 / (2*(floor(4*M_vert))/(first_wedge_endpoint_vert - 1) - 1);
        modif_XX = XX;        % modified to avoid divisions by zero
        modif_XX((modif_XX - 1)/floor(4*M_horiz) + (YY-1)/floor(4*M_vert) == 2) = ...
            modif_XX((modif_XX - 1)/floor(4*M_horiz) + (YY-1)/floor(4*M_vert) == 2) + 1;
        coord_corner = C1 + C2 * ((modif_XX - 1)/(floor(4*M_horiz)) - (YY - 1)/(floor(4*M_vert))) ./ ...
            (2-((modif_XX - 1)/(floor(4*M_horiz)) + (YY - 1)/(floor(4*M_vert))));
        wl_left = wedgewindow(coord_corner);
        [wl_right,wr_right] = wedgewindow(coord_right);
        data = data .* wl_left .* wr_right;
        data = [data; zeros(2*floor(4*M_vert)+1-length_corner_wedge, width_wedge)];
        data = rot90(data,-(quadrant-1));
        bigdata = zeros(bigN1,bigN2);
        bigdata(loc_1,loc_2) = bigdata(loc_1,loc_2) + data;

        % Folding onto N1-by-N2 matrix by periodicity
        temp_bigdata = bigdata(:,(1:N2)+shift_2);
        temp_bigdata(:,N2-shift_2+(1:shift_2)) = temp_bigdata(:,N2-shift_2+(1:shift_2)) + bigdata(:,1:shift_2);
        temp_bigdata(:,1:shift_2) = temp_bigdata(:,1:shift_2) + bigdata(:,N2+shift_2+(1:shift_2));
        bigdata = temp_bigdata((1:N1)+shift_1,:);
        bigdata(N1-shift_1+(1:shift_1),:) = bigdata(N1-shift_1+(1:shift_1),:) + temp_bigdata(1:shift_1,:);
        bigdata(1:shift_1,:) = bigdata(1:shift_1,:) + temp_bigdata(N1+shift_1+(1:shift_1),:);
                % Invariant: size(bigdata) == [N1, N2]
        switch is_real
            case 0
                C{j}{l} = ifft2(ifftshift(bigdata))*sqrt(prod(size(bigdata)));
            case 1
                x = ifft2(ifftshift(bigdata))*sqrt(prod(size(bigdata)));
                C{j}{l} = zeros(2,size(x,1),size(x,2));
                C{j}{l}(1,:,:) = sqrt(2)*real(x);
                C{j}{l}(2,:,:) = sqrt(2)*imag(x);
        end;
                
        % Regular wedges
        length_wedge = floor(4*M_vert) - floor(M_vert);
        Y = 1:length_wedge;
        YY = Y'*ones(1,width_wedge);
        data = Xhi(Y,1:width_wedge);
        temp_XX = XX(Y,1:width_wedge);
        for subl = 2:(nbangles_perquad-1);
            l = l+1;
            wedge_data = data;
            slope_wedge_left = ((floor(4*M_horiz)+1) - wedge_midpoints(subl-1))/floor(4*M_vert);
            mid_line_left = wedge_midpoints(subl-1) + slope_wedge_left*(YY - 1);
            coord_left = 1/2 + floor(4*M_vert)/(wedge_endpoints(subl) - wedge_endpoints(subl-1)) * ...
                (temp_XX - mid_line_left)./(floor(4*M_vert)+1 - YY);
            slope_wedge_right = ((floor(4*M_horiz)+1) - wedge_midpoints(subl))/floor(4*M_vert);
            mid_line_right = wedge_midpoints(subl) + slope_wedge_right*(YY - 1);
            coord_right = 1/2 + floor(4*M_vert)/(wedge_endpoints(subl+1) - wedge_endpoints(subl)) * ...
                (temp_XX - mid_line_right)./(floor(4*M_vert)+1 - YY);
            wl_left = wedgewindow(coord_left);
            [wl_right,wr_right] = wedgewindow(coord_right);            
            wedge_data = wedge_data .* wl_left .* wr_right;
            wedge_data = [wedge_data; zeros(2*floor(4*M_vert)+1-length_wedge, width_wedge)];
            wedge_data = rot90(wedge_data,-(quadrant-1));
            bigdata = zeros(bigN1,bigN2);
            bigdata(loc_1,loc_2) = bigdata(loc_1,loc_2) + wedge_data;
                        
            % Folding onto N1-by-N2 matrix by periodicity
            temp_bigdata = bigdata(:,(1:N2)+shift_2);
            temp_bigdata(:,N2-shift_2+(1:shift_2)) = temp_bigdata(:,N2-shift_2+(1:shift_2)) + bigdata(:,1:shift_2);
            temp_bigdata(:,1:shift_2) = temp_bigdata(:,1:shift_2) + bigdata(:,N2+shift_2+(1:shift_2));
            bigdata = temp_bigdata((1:N1)+shift_1,:);
            bigdata(N1-shift_1+(1:shift_1),:) = bigdata(N1-shift_1+(1:shift_1),:) + temp_bigdata(1:shift_1,:);
            bigdata(1:shift_1,:) = bigdata(1:shift_1,:) + temp_bigdata(N1+shift_1+(1:shift_1),:);
                    % Invariant: size(bigdata) == [N1, N2]
            switch is_real
                case 0
                    C{j}{l} = ifft2(ifftshift(bigdata))*sqrt(prod(size(bigdata)));
                case 1
                    x = ifft2(ifftshift(bigdata))*sqrt(prod(size(bigdata)));
                    C{j}{l} = zeros(2,size(x,1),size(x,2));
                    C{j}{l}(1,:,:) = sqrt(2)*real(x);
                    C{j}{l}(2,:,:) = sqrt(2)*imag(x);
            end;
                        
        end;

        % Right corner wedge
        l = l+1;
        data = Xhi(Y_corner,1:width_wedge);
        YY = Y_corner'*ones(1,width_wedge);
        slope_wedge_left = ((floor(4*M_horiz)+1) - wedge_midpoints(end))/floor(4*M_vert);
        mid_line_left = wedge_midpoints(end) + slope_wedge_left*(YY - 1);
        coord_left = 1/2 + floor(4*M_vert)/(wedge_endpoints(end) - wedge_endpoints(end-1)) * ...
            (XX - mid_line_left)./(floor(4*M_vert)+1 - YY);
        C2 = -1/(2*(floor(4*M_horiz))/(wedge_endpoints(end) - 1) - 1 + 1/(2*(floor(4*M_vert))/(first_wedge_endpoint_vert - 1) - 1));
        C1 = -C2 * (2*(floor(4*M_horiz))/(wedge_endpoints(end) - 1) - 1);
        modif_XX = XX;
        modif_XX((modif_XX - 1)/floor(4*M_horiz) == (YY-1)/floor(4*M_vert)) = ...
            modif_XX((modif_XX - 1)/floor(4*M_horiz) == (YY-1)/floor(4*M_vert)) - 1;
        coord_corner = C1 + C2 * (2-((modif_XX - 1)/(floor(4*M_horiz)) + (YY - 1)/(floor(4*M_vert)))) ./ ...
            ((modif_XX - 1)/(floor(4*M_horiz)) - (YY - 1)/(floor(4*M_vert)));
        wl_left = wedgewindow(coord_left);
        [wl_right,wr_right] = wedgewindow(coord_corner);
        data = data .* wl_left .* wr_right;
        data = [data; zeros(2*floor(4*M_vert)+1-length_corner_wedge, width_wedge)];
        data = rot90(data,-(quadrant-1));
        bigdata = zeros(bigN1,bigN2);
        bigdata(loc_1,loc_2) = bigdata(loc_1,loc_2) + data;
        
        % Folding onto N1-by-N2 matrix by periodicity
        temp_bigdata = bigdata(:,(1:N2)+shift_2);
        temp_bigdata(:,N2-shift_2+(1:shift_2)) = temp_bigdata(:,N2-shift_2+(1:shift_2)) + bigdata(:,1:shift_2);
        temp_bigdata(:,1:shift_2) = temp_bigdata(:,1:shift_2) + bigdata(:,N2+shift_2+(1:shift_2));
        bigdata = temp_bigdata((1:N1)+shift_1,:);
        bigdata(N1-shift_1+(1:shift_1),:) = bigdata(N1-shift_1+(1:shift_1),:) + temp_bigdata(1:shift_1,:);
        bigdata(1:shift_1,:) = bigdata(1:shift_1,:) + temp_bigdata(N1+shift_1+(1:shift_1),:);
                % Invariant: size(bigdata) == [N1, N2]
        switch is_real
            case 0
                C{j}{l} = ifft2(ifftshift(bigdata))*sqrt(prod(size(bigdata)));
            case 1
                x = ifft2(ifftshift(bigdata))*sqrt(prod(size(bigdata)));
                C{j}{l} = zeros(2,size(x,1),size(x,2));
                C{j}{l}(1,:,:) = sqrt(2)*real(x);
                C{j}{l}(2,:,:) = sqrt(2)*imag(x);
        end;
                        
        % Prepare for loop reentry or exit
        if quadrant < nbquadrants, Xhi = rot90(Xhi); end;

    end;    % for quadrant
    
    Xj_topleft_1 = Xj_topleft_1 + floor(4*M1) - floor(2*M1);
    Xj_topleft_2 = Xj_topleft_2 + floor(4*M2) - floor(2*M2);

end;    % for j

% Coarsest wavelet level
M1 = M1/2;
M2 = M2/2;
loc_1 = Xj_topleft_1 + (0:(2*floor(4*M1)));
loc_2 = Xj_topleft_2 + (0:(2*floor(4*M2)));
bigdata = zeros(bigN1,bigN2);
bigdata(loc_1,loc_2) = bigdata(loc_1,loc_2) + Xlow;
% Folding onto N1-by-N2 matrix by periodicity
temp_bigdata = bigdata(:,(1:N2)+shift_2);
temp_bigdata(:,N2-shift_2+(1:shift_2)) = temp_bigdata(:,N2-shift_2+(1:shift_2)) + bigdata(:,1:shift_2);
temp_bigdata(:,1:shift_2) = temp_bigdata(:,1:shift_2) + bigdata(:,N2+shift_2+(1:shift_2));
bigdata = temp_bigdata((1:N1)+shift_1,:);
bigdata(N1-shift_1+(1:shift_1),:) = bigdata(N1-shift_1+(1:shift_1),:) + temp_bigdata(1:shift_1,:);
bigdata(1:shift_1,:) = bigdata(1:shift_1,:) + temp_bigdata(N1+shift_1+(1:shift_1),:);
        % Invariant: size(bigdata) == [N1, N2]

C{nbscales}{1} = ifft2(ifftshift(bigdata))*sqrt(prod(size(bigdata)));
if is_real, C{nbscales}{1} = real(C{nbscales}{1}); end;
%keyboard

% Additional information
C{end} = [N1, N2, is_real];
