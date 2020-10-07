function D = SoftThresh(C,thresh,power)

% SoftThresh.m - Curvelet soft-thresholding
%
% Inputs
%   C           Cell array containing curvelet coefficients coming from
%               RCT.m, using the option is_real = 1.
%   thresh      Amount to which the curvelet shrinkage will be
%               proportional. The actual value of the threshold also
%               depends on the size of the original image (through Donoho's
%               sqrt(2*log(N))) and is proportional to the L^2 norm of each
%               tight frame element.
%   power       Scalar: in Fourier, Log(amplitude noise)/Log(radius). For white
%               noise, power = 0 [default]. For Radon noise, power = 1/2.
%
% Outputs
%   D           Cell array containing thresholded curvelet coefficients
%
% See also RCT.m, IRCT.m
%
% Copyright (c) Laurent Demanet, 2004

nbscales = length(C) - 1;
nbangles_coarse = length(C{end-2});
nbangles = [nbangles_coarse .* 2.^(ceil((nbscales-(2:nbscales))/2)), 1];
N1 = C{end}(1);
N2 = C{end}(2);
is_real = C{end}(3);

% Determining the L^2 norm of each frame element
L2norm = cell(1,nbscales);
for j = 1:(nbscales-1)
    L2norm{j} = cell(1,nbangles(j));
end;

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
        width_wedge = 2*floor(4*M_horiz)+1;
        XX = meshgrid(1:width_wedge,Y_corner);
        YY = Y_corner'*ones(1,width_wedge);
        slope_wedge_right = (floor(4*M_horiz)+1 - wedge_midpoints(1))/floor(4*M_vert);
        mid_line_right = wedge_midpoints(1) + slope_wedge_right*(YY - 1);
                                                            % not integers
                                                            % in general
        coord_right = 1/2 + floor(4*M_vert)/(wedge_endpoints(2) - wedge_endpoints(1)) * ...
            (XX - mid_line_right)./(floor(4*M_vert)+1 - YY);
        C2 = 1/(1/(2*(floor(4*M_horiz))/(wedge_endpoints(1) - 1) - 1) + 1/(2*(floor(4*M_vert))/(first_wedge_endpoint_vert - 1) - 1));
        C1 = C2 / (2*(floor(4*M_vert))/(first_wedge_endpoint_vert - 1) - 1);
        modif_XX = XX;
        modif_XX((modif_XX - 1)/floor(4*M_horiz) + (YY-1)/floor(4*M_vert) == 2) = ...
            modif_XX((modif_XX - 1)/floor(4*M_horiz) + (YY-1)/floor(4*M_vert) == 2) + 1;
        coord_corner = C1 + C2 * ((modif_XX - 1)/(floor(4*M_horiz)) - (YY - 1)/(floor(4*M_vert))) ./ ...
            (2-((modif_XX - 1)/(floor(4*M_horiz)) + (YY - 1)/(floor(4*M_vert))));
        wl_left = wedgewindow(coord_corner);
        [wl_right,wr_right] = wedgewindow(coord_right);

        data = ones(length(Y_corner),width_wedge)/sqrt(N1*N2)/sqrt(2) .* wl_left .* wr_right;
        Xj = zeros(2*floor(4*M1)+1,2*floor(4*M2)+1);
        Xj(Y_corner,1:width_wedge) = Xj(Y_corner,1:width_wedge) + data;
        Xj = Xj .* lowpass;
        Xj_index1 = ((-floor(2*M1)):floor(2*M1)) + floor(4*M1) + 1;
        Xj_index2 = ((-floor(2*M2)):floor(2*M2)) + floor(4*M2) + 1;
        Xj(Xj_index1, Xj_index2) = Xj(Xj_index1, Xj_index2) .* hipass;
        L2norm{j}{l} = sqrt(2*sum(sum(abs(Xj.^2))));
    
        % Regular wedges
        length_wedge = floor(4*M_vert) - floor(M_vert);
        Y = 1:length_wedge;
        YY = Y'*ones(1,width_wedge);
        temp_XX = XX(Y,1:width_wedge);
        for subl = 2:(nbangles_perquad-1);
            l = l+1;
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
            
            data = ones(length(Y),width_wedge)/sqrt(N1*N2)/sqrt(2) .* wl_left .* wr_right;
            Xj = zeros(2*floor(4*M1)+1,2*floor(4*M2)+1);
            Xj(Y,1:width_wedge) = Xj(Y,1:width_wedge) + data;
            Xj = Xj .* lowpass;
            Xj_index1 = ((-floor(2*M1)):floor(2*M1)) + floor(4*M1) + 1;
            Xj_index2 = ((-floor(2*M2)):floor(2*M2)) + floor(4*M2) + 1;
            Xj(Xj_index1, Xj_index2) = Xj(Xj_index1, Xj_index2) .* hipass;
            L2norm{j}{l} = sqrt(2*sum(sum(abs(Xj.^2))));
            
        end;    % for subl
        
        % Right corner wedge
        l = l+1;
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
        
        data = ones(length(Y_corner),width_wedge)/sqrt(N1*N2)/sqrt(2) .* wl_left .* wr_right;
        Xj = zeros(2*floor(4*M1)+1,2*floor(4*M2)+1);;     
        Xj(Y_corner,1:width_wedge) = Xj(Y_corner,1:width_wedge) + data;
        Xj = Xj .* lowpass;
        Xj_index1 = ((-floor(2*M1)):floor(2*M1)) + floor(4*M1) + 1;
        Xj_index2 = ((-floor(2*M2)):floor(2*M2)) + floor(4*M2) + 1;
        Xj(Xj_index1, Xj_index2) = Xj(Xj_index1, Xj_index2) .* hipass;
        L2norm{j}{l} = sqrt(2*sum(sum(abs(Xj.^2))));
   
    end;    % for quadrant
    
    lowpass = lowpass_next;
    
end;    % for j

% Coarsest wavelet level
M1 = M1/2;
M2 = M2/2;

data = ones(2*floor(4*M1)+1,2*floor(4*M2)+1)/sqrt(N1*N2);
data = data .* lowpass;
L2norm{nbscales}{1} = sqrt(sum(sum(abs(data.^2))));

% test L2norm with white noise

%x = randn(N1,N2);
%testC = RCT(x,is_real,nbscales,nbangles_coarse);
%testL2norm = cell(1,nbscales);
%for j = 1:(nbscales-1)
%    testL2norm{j} = cell(1,nbangles(j));
%    for l = 1:nbangles(j)
%       testL2norm{j}{l} = sqrt(sum(sum(sum(testC{j}{l}.^2)))/prod(size(testC{j}{l})));
%   end;
%end;
%testL2norm{nbscales} = cell(1,1);
%testL2norm{nbscales}{1} = sqrt(sum(sum(testC{nbscales}{1}.^2))/prod(size(testC{nbscales}{1})));


% Soft Thresholding

D = C;
scale = 4/3;
for j = 1:(nbscales-1)
    scale = scale/2;
    thresh_j = thresh*sqrt(2*log(N1*N2))*(scale^power);
    for l = 1:nbangles(j)
        thresh_jl = thresh_j*L2norm{j}{l};
        for t = 1:2
            newCjlt = abs(C{j}{l}(t,:,:)) - thresh_jl;
            D{j}{l}(t,:,:) = sign(C{j}{l}(t,:,:)).*newCjlt.*(newCjlt > 0);
        end;
    end;
end;
j = nbscales;
scale = scale/2;
thresh_j = thresh*sqrt(2*log(N1*N2))*(scale^power)*L2norm{j}{1};
newCj1 = abs(C{j}{1}) - thresh_j;
D{j}{1} = sign(C{j}{1}).*newCj1.*(newCj1 > 0);

    

