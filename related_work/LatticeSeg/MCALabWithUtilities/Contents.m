% MCALabWithUtilities Main Directory, Version 110
%
% This is the main directory of the MCALab package with additional transform utilities;
% MCALab is a collection of .m functions for 1D/2D inpainting and separation 
% into morphological components. It is WaveLab compatible and REQUIRES at least WaveLab802
% installed to work properly.
%
%              .m files in this directory
%
%   Contents.m          -    This file
%   MCAPath.m           -    Sets up global variables and pathnames
%   README		-    Short package description.
%   INSTALL		-    Installation instruction.
%   VERSION		-    Version history.
%   THANKS
%   
%
%              Subdirectories
%
%   MCALab110       	-    1D/2D MCA.
%   UDWT                -    UDWT RWT package (DSP Rice). 
%			     Mex files have been compiled for Linux/Unix Solaris/MACOSX and Windows platforms.
%   CurveletToolbox	-    Fast curvelet transform toolbox (Candes et al. curvelet.org).
%			     Two fast curvelet transform codes are included:
%				+ CURVWRAP: Mex Wrapping code. Mex files have been compiled for Linux/Unix Solaris/MACOSX platforms.
%			         If you are running Windows, please compile your own mex. Note that fdct_wrapping.m and ifdct_wrapping.m 
%				 have been slightly modified to handel curvelets at all scales, and adapted to MCALab dictionary structure.
%				+ CURV: Matlab code without any mex file.
%			     
%   Utils		-    Additional provided utilities.
%
%
% N.B.: MCALab incorporates software for two other transforms not distributed with WaveLab. For instance the wrapping 
% version of the Fast Discrete Curvelet Transform implemented in CurveLab, and the UDWT from the RWT toolbox. 
% Note that some of the CurveLab functions have been slightly modified to match our dictionary data structure, 
% and to implement curvelets at the finest scale. We strongly recommend that the user downloads our modified version 
% included in MCALab. Both of these transforms are are in the subdirectories MCALabWithUtilities/CurveleToolbox and
% MCALabWithUtilities/UDWT. The user may read Contents.m in MCALabWithUtilities/ for further details.
% The user is invited to read carefully the license agreement details of these transforms softwares on their respective 
% websites prior to any use.
%
%
% References:
%  J.-L. Starck, M. Elad, and D.L. Donoho. Redundant multiscale transforms and
%  their application for morphological component analysis. Advances in Imaging
%  and Electron Physics, 132, 2004.
% 
%  M. Elad, J.-L Starck, D. Donoho and P. Querre,  "Simultaneous Cartoon and
%  Texture Image Inpainting using Morphological Component Analysis (MCA)", ACHA.
%  To appear.
%
%  M.J. Fadili and J.-L. Starck. Em algorithm for sparse representation-based
%  image inpainting. In IEEE Intl. conference on Image Signal Processing, Genoa,
%  Italy, Sept. 2005.
%
%  M.J. Fadili, J.-L. Starck and F. Murtagh. Inpainting and Zooming using Sparse 
%  Representations,  The Computer Journal, in press, 2007.
%
%
%
% Copyright (c) 2004. Jalal M. Fadili/CNRS
% Copyright (c) 2006. Jalal M. Fadili/CNRS
% Copyright (c) 2008. Jalal M. Fadili/CNRS
%     
%   
% Part of MCALab Version 110
% Built March 2004
% This material is distributed under the CECILL licence
% For Copying permissions see LICENSE.CeCILL
% Comments? e-mail Jalal@Fadili.greyc.ensicaen.fr
% 
