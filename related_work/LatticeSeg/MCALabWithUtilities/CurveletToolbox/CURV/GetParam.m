function [Angles, Xlocations, Ylocations] = GetParam(C)

% GetParam.m - Obtain the angle and location relative to each curvelet
% coefficient
%
% Inputs
%   C           Cell array C{j}{l}(k1,k2) containing curvelet
%               coefficients (see description in FCT.m)
%
% Outputs
%   Angles      Cell array Angles{j}(l) giving the angle, in degrees,
%               between the x-axis and the codirection of each curvelet
%               indexed by j and l (angle increases counterclockwise from 0
%               to 180, or from 0 to 360).
%   XLocations  Cell array XLocations{j}{l}(k1,k2) giving the x coordinate,
%               on the original M-by-N image grid, of the curvelet indexed
%               by j, l, k1 and k2 (x increases from left to right, from 1
%               to N).
%   YLocations  Same for the y coordinate (y increases from top to bottom,
%               from 1 to M).
%
% See also FCT.m, FICT.m
%
% Copyright (c) Laurent Demanet, 2004

% Initialization
nbscales = length(C) - 1;
nbangles_coarse = length(C{end-2});
is_real = C{end}(3);
nbangles = [nbangles_coarse .* 2.^(ceil((nbscales-(2:nbscales))/2)), 1];
N1 = C{end}(1);
N2 = C{end}(2);
M1 = N1/3;
M2 = N2/3;
Angles = cell(1,nbscales);
for j = 1:nbscales
    Angles{j} = zeros(1,nbangles(j));
end;
Xlocations = cell(1,nbscales);
for j = 1:nbscales
    Xlocations{j} = cell(1,nbangles(j));
end;
Ylocations = cell(1,nbscales);
for j = 1:nbscales
    Ylocations{j} = cell(1,nbangles(j));
end;

for j = 1:(nbscales-1),

    M1 = M1/2;
    M2 = M2/2;
   
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
        
        % Left corner wedge
        
        l = l+1;
        first_wedge_endpoint_vert = round(2*floor(4*M_vert)/(2*nbangles_perquad) + 1);
        length_corner_wedge = floor(4*M_vert) - floor(M_vert) + ceil(first_wedge_endpoint_vert/4);
        width_wedge = wedge_endpoints(2) + wedge_endpoints(1) - 1;
        slope_wedge = (floor(4*M_horiz) + 1 - wedge_endpoints(1))/floor(4*M_vert);

        if is_real == 1, length_corner_wedge = 2*length_corner_wedge+1; end;
        size_wedge_horiz = width_wedge * (mod(quadrant,2)==1) + length_corner_wedge * (mod(quadrant,2)==0);
        size_wedge_vert = width_wedge * (mod(quadrant,2)==0) + length_corner_wedge * (mod(quadrant,2)==1);
        Angles{j}(l) = mod(pi/2 + atan(slope_wedge) - pi/2*(quadrant-1), 2*pi)*180/pi;
        xloc = 1 + N2*(0:(1/size_wedge_horiz):(1-1/size_wedge_horiz));
        Xlocations{j}{l} = ones(size_wedge_vert,1) * xloc;
        yloc = 1 + N1*(0:(1/size_wedge_vert):(1-1/size_wedge_vert));
        Ylocations{j}{l} = yloc' * ones(1,size_wedge_horiz);
        
        % Regular wedges
        length_wedge = floor(4*M_vert) - floor(M_vert);
        if is_real == 1, length_wedge = 2*length_wedge+1; end;
        for subl = 2:(nbangles_perquad-1);
            l = l+1;
            width_wedge = wedge_endpoints(subl+1) - wedge_endpoints(subl-1) + 1;
            slope_wedge = ((floor(4*M_horiz)+1) - wedge_endpoints(subl))/floor(4*M_vert);

            size_wedge_horiz = width_wedge * (mod(quadrant,2)==1) + length_wedge * (mod(quadrant,2)==0);
            size_wedge_vert = width_wedge * (mod(quadrant,2)==0) + length_wedge * (mod(quadrant,2)==1);
            Angles{j}(l) = mod(pi/2 + atan(slope_wedge) - pi/2*(quadrant-1), 2*pi)*180/pi;
            xloc = 1 + N2*(0:(1/size_wedge_horiz):(1-1/size_wedge_horiz));
            Xlocations{j}{l} = ones(size_wedge_vert,1) * xloc;
            yloc = 1 + N1*(0:(1/size_wedge_vert):(1-1/size_wedge_vert));
            Ylocations{j}{l} = yloc' * ones(1,size_wedge_horiz);
        
        end;    % for subl
        
        % Right corner wedge
        l = l+1;
        width_wedge = 4*floor(4*M_horiz) + 3 - wedge_endpoints(end) - wedge_endpoints(end-1);
        slope_wedge = ((floor(4*M_horiz)+1) - wedge_endpoints(end))/floor(4*M_vert);
        
        size_wedge_horiz = width_wedge * (mod(quadrant,2)==1) + length_corner_wedge * (mod(quadrant,2)==0);
        size_wedge_vert = width_wedge * (mod(quadrant,2)==0) + length_corner_wedge * (mod(quadrant,2)==1);
        Angles{j}(l) = mod(pi/2 + atan(slope_wedge) - pi/2*(quadrant-1), 2*pi)*180/pi;
        xloc = 1 + N2*(0:(1/size_wedge_horiz):(1-1/size_wedge_horiz));
        Xlocations{j}{l} = ones(size_wedge_vert,1) * xloc;
        yloc = 1 + N1*(0:(1/size_wedge_vert):(1-1/size_wedge_vert));
        Ylocations{j}{l} = yloc' * ones(1,size_wedge_horiz);
        
    end;    % for quadrant
    
end;    % for j

% Coarsest wavelet level
M1 = M1/2;
M2 = M2/2;
[len, width] = size(C{nbscales}{1});
Angles{nbscales} = 0;
xloc = 1 + N2*(0:(1/width):(1-1/width));
Xlocations{nbscales}{1} = ones(len,1) * xloc;
yloc = 1 + N1*(0:(1/len):(1-1/len));
Ylocations{nbscales}{1} = yloc' * ones(1,width);