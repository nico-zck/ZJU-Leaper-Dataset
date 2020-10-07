function x = FICT(C)

% FICT.m - Fast Inverse Curvelet Transform
% This is in fact the adjoint, also the pseudo-inverse
%
% Inputs
%   C           Cell array containing curvelet coefficients (see
%               description in FCT.m)
%
% Outputs
%   x           M-by-N matrix
%
% See also FCT.m
%
% Copyright (c) Laurent Demanet, 2004

% Initialization
nbscales = length(C) - 1;
nbangles_coarse = length(C{end-2});
nbangles = [nbangles_coarse .* 2.^(ceil((nbscales-(2:nbscales))/2)), 1];
N1 = C{end}(1);
N2 = C{end}(2);
is_real = C{end}(3);
M1 = N1/3;
M2 = N2/3;
bigN1 = 2*floor(2*M1)+1;
bigN2 = 2*floor(2*M2)+1;
X = zeros(bigN1,bigN2);

% Initialization: preparing the lowpass filter at finest scale
window_length_1 = floor(2*M1) - floor(M1) - 1 - (mod(N1,3)==0);
window_length_2 = floor(2*M2) - floor(M2) - 1 - (mod(N2,3)==0);
coord_1 = 0:(1/window_length_1):1;
coord_2 = 0:(1/window_length_2):1;
[wl_1,wr_1] = wedgewindow(coord_1);
[wl_2,wr_2] = wedgewindow(coord_2);
lowpass_1 = [wl_1, ones(1,2*floor(M1)+1), wr_1];
if mod(N1,3)==0, lowpass_1 = [0, lowpass_1, 0]; end;
lowpass_2 = [wl_2, ones(1,2*floor(M2)+1), wr_2];
if mod(N2,3)==0, lowpass_2 = [0, lowpass_2, 0]; end;
lowpass = lowpass_1'*lowpass_2;

% Loop: pyramidal reconstruction
Xj_topleft_1 = 1;
Xj_topleft_2 = 1;
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
    lowpass_next = lowpass_1'*lowpass_2;
    hipass = sqrt(1 - lowpass_next.^2);
    Xj = zeros(2*floor(4*M1)+1,2*floor(4*M2)+1);
    
    % Loop: angles
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
        
        % Left corner wedge
        
        l = l+1;
        first_wedge_endpoint_vert = round(2*floor(4*M_vert)/(2*nbangles_perquad) + 1);
        length_corner_wedge = floor(4*M_vert) - floor(M_vert) + ceil(first_wedge_endpoint_vert/4);
        Y_corner = 1:length_corner_wedge;
        XX = meshgrid(1:(2*floor(4*M_horiz)+1),Y_corner);
        width_wedge = wedge_endpoints(2) + wedge_endpoints(1) - 1;
        slope_wedge = (floor(4*M_horiz) + 1 - wedge_endpoints(1))/floor(4*M_vert);
        left_line = round(2 - wedge_endpoints(1) + slope_wedge*(Y_corner - 1));
        wrapped_XX = zeros(length_corner_wedge,width_wedge);
        for row = Y_corner
            cols = left_line(row) + mod((0:(width_wedge-1))-(left_line(row)-left_line(1)),width_wedge);
            admissible_cols = round(1/2*(cols+1+abs(cols-1)));
            wrapped_XX(row,:) = XX(row,admissible_cols);
        end;
        YY = Y_corner'*ones(1,width_wedge);
        slope_wedge_right = (floor(4*M_horiz)+1 - wedge_midpoints(1))/floor(4*M_vert);
        mid_line_right = wedge_midpoints(1) + slope_wedge_right*(YY - 1);
                                                            % not integers
                                                            % in general
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
        switch is_real
            case 0
                wrapped_data = fft2(C{j}{l})/sqrt(prod(size(C{j}{l})));
                wrapped_data = rot90(wrapped_data,(quadrant-1));
            case 1
                x = fft2(C{j}{l})/sqrt(prod(size(C{j}{l})));
                x = rot90(x,(quadrant-1));
                wrapped_data = x(1:((end-1)/2),:);
            case 2
                x = squeeze(C{j}{l}(1,:,:)) + sqrt(-1)*squeeze(C{j}{l}(2,:,:));
                wrapped_data = fft2(x)/sqrt(prod(size(x)))/sqrt(2);
                wrapped_data = rot90(wrapped_data,(quadrant-1));
        end;
        wrapped_data = wrapped_data .* wl_left .* wr_right;

        % Unwrapping data
        for row = Y_corner
            cols = left_line(row) + mod((0:(width_wedge-1))-(left_line(row)-left_line(1)),width_wedge);
            admissible_cols = round(1/2*(cols+1+abs(cols-1)));
            Xj(row,admissible_cols) = Xj(row,admissible_cols) + wrapped_data(row,:);
                                % We use the following property: in an assignment
                                % A(B) = C where B and C are vectors, if
                                % some value x repeats in B, then the
                                % last occurrence of x is the one
                                % corresponding to the eventual assignment.
        end;

        % Regular wedges
        length_wedge = floor(4*M_vert) - floor(M_vert);
        Y = 1:length_wedge;
        for subl = 2:(nbangles_perquad-1);
            l = l+1;
            width_wedge = wedge_endpoints(subl+1) - wedge_endpoints(subl-1) + 1;
            slope_wedge = ((floor(4*M_horiz)+1) - wedge_endpoints(subl))/floor(4*M_vert);
            left_line = round(wedge_endpoints(subl-1) + slope_wedge*(Y - 1));
            wrapped_XX = zeros(length_wedge,width_wedge);
            for row = Y
                cols = left_line(row) + mod((0:(width_wedge-1))-(left_line(row)-left_line(1)),width_wedge);
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
            switch is_real
                case 0
                    wrapped_data = fft2(C{j}{l})/sqrt(prod(size(C{j}{l})));
                    wrapped_data = rot90(wrapped_data,(quadrant-1));
                case 1
                    x = fft2(C{j}{l})/sqrt(prod(size(C{j}{l})));
                    x = rot90(x,(quadrant-1));
                    wrapped_data = x(1:((end-1)/2),:);
                case 2
                    x = squeeze(C{j}{l}(1,:,:)) + sqrt(-1)*squeeze(C{j}{l}(2,:,:));
                    wrapped_data = fft2(x)/sqrt(prod(size(x)))/sqrt(2);
                    wrapped_data = rot90(wrapped_data,(quadrant-1));
            end;
            wrapped_data = wrapped_data .* wl_left .* wr_right;
            
            % Unwrapping data
            for row = Y
                cols = left_line(row) + mod((0:(width_wedge-1))-(left_line(row)-left_line(1)),width_wedge);
                Xj(row,cols) = Xj(row,cols) + wrapped_data(row,:);
            end;

        end;    % for subl
        
        % Right corner wedge
        l = l+1;
        width_wedge = 4*floor(4*M_horiz) + 3 - wedge_endpoints(end) - wedge_endpoints(end-1);
        slope_wedge = ((floor(4*M_horiz)+1) - wedge_endpoints(end))/floor(4*M_vert);
        left_line = round(wedge_endpoints(end-1) + slope_wedge*(Y_corner - 1));
        wrapped_XX = zeros(length_corner_wedge,width_wedge);
        for row = Y_corner
            cols = left_line(row) + mod((0:(width_wedge-1))-(left_line(row)-left_line(1)),width_wedge);
            admissible_cols = round(1/2*(cols+2*floor(4*M_horiz)+1-abs(cols-(2*floor(4*M_horiz)+1))));
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
        switch is_real
            case 0
                wrapped_data = fft2(C{j}{l})/sqrt(prod(size(C{j}{l})));
                wrapped_data = rot90(wrapped_data,(quadrant-1));
            case 1
                x = fft2(C{j}{l})/sqrt(prod(size(C{j}{l})));
                x = rot90(x,(quadrant-1));
                wrapped_data = x(1:((end-1)/2),:);
            case 2
                x = squeeze(C{j}{l}(1,:,:)) + sqrt(-1)*squeeze(C{j}{l}(2,:,:));
                wrapped_data = fft2(x)/sqrt(prod(size(x)))/sqrt(2);
                wrapped_data = rot90(wrapped_data,(quadrant-1));
        end;
        wrapped_data = wrapped_data .* wl_left .* wr_right;
        
         % Unwrapping data
        for row = Y_corner
            cols = left_line(row) + mod((0:(width_wedge-1))-(left_line(row)-left_line(1)),width_wedge);
            admissible_cols = round(1/2*(cols+2*floor(4*M_horiz)+1-abs(cols-(2*floor(4*M_horiz)+1))));
            Xj(row,fliplr(admissible_cols)) = Xj(row,fliplr(admissible_cols)) + wrapped_data(row,end:-1:1);
                                % We use the following property: in an assignment
                                % A(B) = C where B and C are vectors, if
                                % some value x repeats in B, then the
                                % last occurrence of x is the one
                                % corresponding to the eventual assignment.
        end;

        Xj = rot90(Xj);
        
    end;    % for quadrant
    
    Xj = Xj .* lowpass;
    Xj_index1 = ((-floor(2*M1)):floor(2*M1)) + floor(4*M1) + 1;
    Xj_index2 = ((-floor(2*M2)):floor(2*M2)) + floor(4*M2) + 1;
    Xj(Xj_index1, Xj_index2) = Xj(Xj_index1, Xj_index2) .* hipass;
    loc_1 = Xj_topleft_1 + (0:(2*floor(4*M1)));
    loc_2 = Xj_topleft_2 + (0:(2*floor(4*M2)));
    X(loc_1,loc_2) = X(loc_1,loc_2) + Xj;

    % Preparing for loop reentry or exit
    Xj_topleft_1 = Xj_topleft_1 + floor(4*M1) - floor(2*M1);
    Xj_topleft_2 = Xj_topleft_2 + floor(4*M2) - floor(2*M2);
    lowpass = lowpass_next;
    
end;    % for j

if (~~is_real)
    Y = X;
    X = rot90(X,2);
    X = X + conj(Y);
end
    
% Coarsest wavelet level
M1 = M1/2;
M2 = M2/2;
Xj = fftshift(fft2(C{nbscales}{1}))/sqrt(prod(size(C{nbscales}{1})));
loc_1 = Xj_topleft_1 + (0:(2*floor(4*M1)));
loc_2 = Xj_topleft_2 + (0:(2*floor(4*M2)));
X(loc_1,loc_2) = X(loc_1,loc_2) + Xj .* lowpass;

% Folding back onto N1-by-N2 matrix
M1 = N1/3;
M2 = N2/3;
shift_1 = floor(2*M1)-floor(N1/2);
shift_2 = floor(2*M2)-floor(N2/2);
Y = X(:,(1:N2)+shift_2);
Y(:,N2-shift_2+(1:shift_2)) = Y(:,N2-shift_2+(1:shift_2)) + X(:,1:shift_2);
Y(:,1:shift_2) = Y(:,1:shift_2) + X(:,N2+shift_2+(1:shift_2));
X = Y((1:N1)+shift_1,:);
X(N1-shift_1+(1:shift_1),:) = X(N1-shift_1+(1:shift_1),:) + Y(1:shift_1,:);
X(1:shift_1,:) = X(1:shift_1,:) + Y(N1+shift_1+(1:shift_1),:);

x = ifft2(ifftshift(X))*sqrt(prod(size(X)));
if ~~is_real, x = real(x); end;


