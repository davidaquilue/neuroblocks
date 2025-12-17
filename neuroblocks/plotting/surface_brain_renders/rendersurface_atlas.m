function rendersurface_atlas( ...
    atlasName, ...
    parcelValues, ...
    outputDir, ...
    fileRoot, ...
    rangemin, ...
    rangemax, ...
    inv, ...
    clmap, ...
    surfacetype, ...
    titletext, ...
    plotflats ...
    )
% RENDERSURFACE_ATLAS
% Generic cortical surface renderer for parcel-wise atlas data
%
% atlasName     : string, descriptive name of atlas (e.g. 'DBS80')
% parcelValues  : vector of parcel-wise scalar values
% outputDir     : directory where to store the figure
%
% Optional:
% fileRoot          : file_name root for the figure (default: 4views)
% rangemin, rangemax: color limits
% inv               : colormap mode
% clmap             : colormap name
% surfacetype       : 1=mid, 2=inflated, 3=very inflated
% titletext         : string to plot as title (not too clean)
% plotflats         : (bool) whether to plot flattened cortical surfaces

%% -------------------- defaults --------------------

if ~exist('fileRoot','var') || isempty(fileRoot)
    fileRoot = "figure";
end


if ~exist('rangemin','var') || isempty(rangemin)
    rangemin = min(parcelValues);
end

if ~exist('rangemax','var') || isempty(rangemax)
    rangemax = max(parcelValues);
end

if ~exist('inv','var') || isempty(inv)
    inv = 0;
end

if ~exist('clmap','var') || isempty(clmap)
    clmap = 'BrBG5';
end

if ~exist('surfacetype','var') || isempty(surfacetype)
    surfacetype = 2;
end

if ~exist('plotflats','var') || isempty(plotflats)
    plotflats = 0;
end

if atlasName == "DesikanKilliany" || atlasName == "DBS80"
    error("Rendering Function not yet available for DesikanKilliany and DBS80")
end

%% -------------------- paths --------------------

thisFile = mfilename('fullpath');
thisDir  = fileparts(thisFile);
atlasdir = char(fullfile(thisDir, "atlas_surfaces"));

%% -------------------- subplot layout --------------------

subplot = @(m,n,p) subtightplot(m,n,p,[0.01 0.05],[0.1 0.01],[0.1 0.01]);

%% -------------------- load surfaces --------------------
% NOTE: surfaces should be atlas-agnostic
fsdir = char(fullfile(atlasdir, "fsaverage"));
surf.L.mid   = gifti([fsdir '/fs_LR.32k.L.midthickness.surf.gii']);
surf.L.infl  = gifti([fsdir '/fs_LR.32k.L.inflated.surf.gii']);
surf.L.vinfl = gifti([fsdir '/fs_LR.32k.L.very_inflated.surf.gii']);
surf.L.flat  = gifti([fsdir '/fs_LR.32k.L.flat.surf.gii']);

surf.R.mid   = gifti([fsdir '/fs_LR.32k.R.midthickness.surf.gii']);
surf.R.infl  = gifti([fsdir '/fs_LR.32k.R.inflated.surf.gii']);
surf.R.vinfl = gifti([fsdir '/fs_LR.32k.R.very_inflated.surf.gii']);
surf.R.flat  = gifti([fsdir '/fs_LR.32k.R.flat.surf.gii']);

%% -------------------- choose display surface --------------------

switch surfacetype
    case 1
        sl = surf.L.mid;
        sr = surf.R.mid;
    case 2
        sl = surf.L.infl;
        sr = surf.R.infl;
    case 3
        sl = surf.L.vinfl;
        sr = surf.R.vinfl;
    otherwise
        error('Unknown surfacetype')
end

%% -------------------- load atlas labels --------------------
% atlasInfo must define label files

if contains(atlasName, "Schaefer")
    N = str2double(strrep(atlasName, "Schaefer", ""));
    label_L = gifti(char(fullfile(atlasdir, "Schaefer", "Schaefer2018_7Networks_" + N + ".32k.L.label.gii")));
    label_R = gifti(char(fullfile(atlasdir, "Schaefer", "Schaefer2018_7Networks_" + N + ".32k.R.label.gii")));
else
    label_L = gifti(char(fullfile(atlasdir, atlasName, atlasName + ".32k.L.label.gii")));
    label_R = gifti(char(fullfile(atlasdir, atlasName, atlasName + ".32k.R.label.gii")));
end

%% -------------------- initialize vertex data --------------------
% Use empty func.gii templates or create zeros manually

vl = zeros(size(sl.vertices, 1), 1);
vr = zeros(size(sr.vertices, 1), 1);

%% -------------------- map parcels to vertices --------------------

% Atlas-specific labels (Desikan and DBS80) have had changes and some don't
% follow the same pattern (e.g. Schaefer 1:N, Glasser L-> 1:N/2, R->1:N/2)

if atlasName == "DesikanKilliany" || atlasName == "DBS80"  % 68 Cortical regions
    labels_l = 1:35; labels_l(4) = [];
    labels_r = 1:35; labels_r(4) = [];
    rh_extra_idx = 34;  % index in parcellation where Right Hemisph count starts

elseif contains(atlasName, "Schaefer")
    N = str2double(strrep(atlasName, "Schaefer", ""));
    labels_l = 1:N/2; labels_r = (N/2 + 1):N;
    rh_extra_idx = N/2;  % In this case, the labels_r already start from high number

elseif atlasName == "Glasser"  % 360 Cortical Regions
    labels_l = 1:180; labels_r = 1:180;
    rh_extra_idx = 180;
else
    error("atlas " + atlasName + " not yet implemented")
end

% Now fill in the maps between parcels and vertices
for i = 1:numel(labels_l)
    lbl = labels_l(i);
    vl(label_L.cdata == lbl) = parcelValues(i);
end

for i = 1:numel(labels_r)
    lbl = labels_r(i);
    parcel_idx = rh_extra_idx + i;
    vr(label_R.cdata == lbl) = parcelValues(parcel_idx);
end

%% -------------------- clean medial wall --------------------

vl(label_L.cdata < 0) = 0;
vr(label_R.cdata < 0) = 0;

%% -------------------- rendering --------------------

fig = figure('Position',[100 100 500 500], 'Visible', 'off');
if plotflats
    nrows = 3;
    ncols = 2;
else
    nrows = 4;
    ncols = 1;
end
% Left lateral
subplot(nrows,ncols,1)
render_patch(sl, vl, rangemin, rangemax, [-90 0])

% Right medial
subplot(nrows,ncols,2)
render_patch(sr, vr, rangemin, rangemax, [90 0])

% Left medial
subplot(nrows,ncols,3)
render_patch(sl, vl, rangemin, rangemax, [90 0])

% Right lateral
subplot(nrows,ncols,4)
render_patch(sr, vr, rangemin, rangemax, [-90 0])

% Flat maps
if plotflats
    subplot(nrows,ncols,5)
    render_patch(surf.L.flat, vl, rangemin, rangemax, [0 90])
    
    subplot(nrows,ncols,6)
    render_patch(surf.R.flat, vr, rangemin, rangemax, [0 90])
end

cb = colorbar('southoutside');
clim([rangemin rangemax]);
if plotflats
    cb.Position = [0.25 0.05 0.5 0.02];  % Adjust these values
else
    cb.Position = [0.4475 0.05 0.2 0.02];   % Narrower for single column
end

%% -------------------- colormap --------------------

switch inv
    case 0
        c = othercolor(clmap);
    case 1
        c = flipud(othercolor(clmap));
    case 2
        c = othercolor(clmap,3);
end

% medial wall color
c(1,:) = [0.95 0.95 0.95];
colormap(c)

if exist('titletext','var') && ~isempty(titletext)
    sgtitle(titletext)

end

exportgraphics(fig, fullfile(outputDir, fileRoot + ".pdf"), 'ContentType', 'vector')
close(fig)
