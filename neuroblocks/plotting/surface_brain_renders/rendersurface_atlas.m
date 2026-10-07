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
    plotflats, ...
    saveAsImg, ...
    parcelAlpha, ...
    parcelOutline, ...
    outlineColor, ...
    outlineWidth, ...
    backgroundColor, ...
    faceColor ...
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
% saveAsImg         : (bool) save as .png instead of vector .pdf
% parcelAlpha       : parcel-wise values in [0 1]; 1 = full color, 0 = background
%                     color. Parcels with NaN value get alpha 0. (default: ones)
% parcelOutline     : parcel-wise logical; true = draw a closed contour around
%                     the parcel (default: false)
% outlineColor      : RGB color of the contours (default: [0 0 0])
% outlineWidth      : line width of the contours (default: 1)
% backgroundColor   : RGB color that faded parcels blend into
%                     (default: [0.85 0.85 0.85])
% faceColor         : patch FaceColor, 'interp' (default) or 'flat'

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
    clmap = 'Cat_12';
end

if ~exist('surfacetype','var') || isempty(surfacetype)
    surfacetype = 2;
end

if ~exist('plotflats','var') || isempty(plotflats)
    plotflats = 0;
end

if ~exist('saveAsImg','var') || isempty(saveAsImg)
    saveAsImg = 1;
end

nParcels = numel(parcelValues);

if ~exist('parcelAlpha','var') || isempty(parcelAlpha)
    parcelAlpha = ones(nParcels, 1);
end

if ~exist('parcelOutline','var') || isempty(parcelOutline)
    parcelOutline = false(nParcels, 1);
end

if ~exist('outlineColor','var') || isempty(outlineColor)
    outlineColor = [0 0 0];
end

if ~exist('outlineWidth','var') || isempty(outlineWidth)
    outlineWidth = 1;
end

if ~exist('backgroundColor','var') || isempty(backgroundColor)
    backgroundColor = [0.85 0.85 0.85];
end

if ~exist('faceColor','var') || isempty(faceColor)
    faceColor = 'interp';
end

if numel(parcelAlpha) ~= nParcels
    error("parcelAlpha has %d elements, expected %d (one per parcel)", numel(parcelAlpha), nParcels)
end

if numel(parcelOutline) ~= nParcels
    error("parcelOutline has %d elements, expected %d (one per parcel)", numel(parcelOutline), nParcels)
end

parcelAlpha = double(parcelAlpha(:));
if any(~(parcelAlpha >= 0 & parcelAlpha <= 1))
    error("parcelAlpha values must be within [0, 1]")
end
parcelAlpha(isnan(parcelValues(:))) = 0;  % missing values drawn as background

parcelOutline = logical(parcelOutline(:));
outlineColor = double(outlineColor(:)');
backgroundColor = double(backgroundColor(:)');

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

%% -------------------- map parcels to vertices --------------------

[pidx_l, pidx_r] = parcel_vertex_index(atlasName, label_L, label_R);

vl = parcel_to_vertex(parcelValues, pidx_l, 0);
vr = parcel_to_vertex(parcelValues, pidx_r, 0);

al = parcel_to_vertex(parcelAlpha, pidx_l, 0);
ar = parcel_to_vertex(parcelAlpha, pidx_r, 0);

kl = parcel_to_vertex(parcelOutline, pidx_l, false);
kr = parcel_to_vertex(parcelOutline, pidx_r, false);

medial_l = label_L.cdata <= 0;
medial_r = label_R.cdata <= 0;

%% -------------------- colormap --------------------

fig = figure('Position',[100 100 500 500], 'Visible', 'off');

switch inv
    case 0
        c = othercolor(clmap);
    case 1
        c = flipud(othercolor(clmap));
    case 2
        c = othercolor(clmap,3);
end

%% -------------------- per-vertex truecolor --------------------
% Colors are computed here instead of through the axes colormap, so that
% the medial wall does not take over an entry of the colormap. Alpha is
% blended manually into backgroundColor: real transparency breaks vector
% export and interacts badly with lighting.

medial_color = [0.95 0.95 0.95];
cl = values_to_rgb(vl, c, rangemin, rangemax);
cr = values_to_rgb(vr, c, rangemin, rangemax);
cl = al .* cl + (1 - al) .* backgroundColor;
cr = ar .* cr + (1 - ar) .* backgroundColor;
cl(medial_l,:) = repmat(medial_color, nnz(medial_l), 1);
cr(medial_r,:) = repmat(medial_color, nnz(medial_r), 1);

%% -------------------- outlines --------------------

outline_l = [];
outline_r = [];
if any(parcelOutline)
    outline_l = struct('indicator', kl, 'color', outlineColor, 'width', outlineWidth);
    outline_r = struct('indicator', kr, 'color', outlineColor, 'width', outlineWidth);
end

%% -------------------- rendering --------------------

if plotflats
    nrows = 3;
    ncols = 2;
else
    nrows = 4;
    ncols = 1;
end
% Left lateral
subplot(nrows,ncols,1)
render_patch(sl, cl, rangemin, rangemax, [-90 0], outline_l, faceColor)

% Right medial
subplot(nrows,ncols,2)
render_patch(sr, cr, rangemin, rangemax, [90 0], outline_r, faceColor)

% Left medial
subplot(nrows,ncols,3)
render_patch(sl, cl, rangemin, rangemax, [90 0], outline_l, faceColor)

% Right lateral
subplot(nrows,ncols,4)
render_patch(sr, cr, rangemin, rangemax, [-90 0], outline_r, faceColor)

% Flat maps
if plotflats
    subplot(nrows,ncols,5)
    render_patch(surf.L.flat, cl, rangemin, rangemax, [0 90], outline_l, faceColor)

    subplot(nrows,ncols,6)
    render_patch(surf.R.flat, cr, rangemin, rangemax, [0 90], outline_r, faceColor)
end

% With outlines, a vector export makes MATLAB sort every triangle of the lit
% surface: huge files that also show contours hidden behind the surface.
% Embed the panels as OpenGL images instead (MATLAB already does so for lit
% surfaces without lines); colorbar and title stay vector.
if ~saveAsImg && any(parcelOutline)
    rasterize_panels(fig, 600)
end

% Patches are truecolor; colormap and clim only drive the colorbar
cb = colorbar('southoutside');
clim([rangemin rangemax]);
if plotflats
    cb.Position = [0.25 0.05 0.5 0.02];  % Adjust these values
else
    cb.Position = [0.4475 0.05 0.2 0.02];   % Narrower for single column
end
colormap(c)

if exist('titletext','var') && ~isempty(titletext)
    sgtitle(titletext)

end

if saveAsImg
    exportgraphics(fig, fullfile(outputDir, fileRoot + ".png"))
else
    exportgraphics(fig, fullfile(outputDir, fileRoot + ".pdf"), 'ContentType', 'vector')
end
close(fig)


function [pidx_l, pidx_r] = parcel_vertex_index(atlasName, label_L, label_R)
% For each vertex, the index into the parcel-wise vectors of the parcel it
% belongs to. 0 means medial wall or no parcel.

pidx_l = zeros(numel(label_L.cdata), 1);
pidx_r = zeros(numel(label_R.cdata), 1);

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

elseif atlasName == "AAL"  % AAL with 90 regions
    % The gifti parcellation has 42 labels (0=medial wall, 1-41=cortical).
    % aalG maps each of the 41 cortical gifti labels to its AAL region
    % pair. AAL odd indices are left, even are right hemisphere.
    % Subcortical regions (AAL 71-78: Caudate, Putamen, Pallidum,
    % Thalamus) are not in the gifti surface parcellation.
    %
    % pos | gifti | AAL L/R | Region
    % ----|-------|---------|-----------------------------------
    %   1 |     1 |    1/ 2 | Precentral gyrus
    %   2 |     2 |    3/ 4 | Superior Frontal gyrus
    %   3 |    13 |    5/ 6 | Superior Frontal gyrus, Orbital
    %   4 |     3 |    7/ 8 | Middle Frontal gyrus
    %   5 |    14 |    9/10 | Middle Frontal gyrus, Orbital
    %   6 |     4 |   11/12 | Inferior Frontal gyrus, Opercular
    %   7 |     5 |   13/14 | Inferior Frontal gyrus, Triangular
    %   8 |     6 |   15/16 | Inferior Frontal gyrus, Orbital
    %   9 |     7 |   17/18 | Rolandic operculum
    %  10 |     8 |   19/20 | Supplementary Motor area
    %  11 |    15 |   21/22 | Olfactory cortex
    %  12 |     9 |   23/24 | Superior Frontal gyrus, Medial
    %  13 |    10 |   25/26 | Superior Frontal gyrus, Medial Orbital
    %  14 |    11 |   27/28 | Gyrus Rectus
    %  15 |    16 |   29/30 | Insula
    %  16 |    17 |   31/32 | Cingulate gyrus, Anterior
    %  17 |    18 |   33/34 | Cingulate gyrus, Middle
    %  18 |    19 |   35/36 | Cingulate gyrus, Posterior
    %  19 |    20 |   37/38 | Hippocampus
    %  20 |    21 |   39/40 | Parahippocampus
    %  21 |    12 |   41/42 | Amygdala
    %  22 |    22 |   43/44 | Calcarine fissure
    %  23 |    23 |   45/46 | Cuneus
    %  24 |    24 |   47/48 | Lingual gyrus
    %  25 |    25 |   49/50 | Superior Occipital lobe
    %  26 |    26 |   51/52 | Middle Occipital lobe
    %  27 |    27 |   53/54 | Inferior Occipital lobe
    %  28 |    28 |   55/56 | Fusiform gyrus
    %  29 |    29 |   57/58 | Postcentral gyrus
    %  30 |    30 |   59/60 | Superior Parietal gyrus
    %  31 |    31 |   61/62 | Inferior Parietal gyrus
    %  32 |    32 |   63/64 | Supramarginal gyrus
    %  33 |    33 |   65/66 | Angular gyrus
    %  34 |    34 |   67/68 | Precuneus
    %  35 |    35 |   69/70 | Paracentral lobule
    %  36 |    36 |   79/80 | Heschl's gyrus
    %  37 |    37 |   81/82 | Superior Temporal gyrus
    %  38 |    38 |   83/84 | Temporal pole, Superior Temporal
    %  39 |    39 |   85/86 | Middle Temporal gyrus
    %  40 |    40 |   87/88 | Temporal pole, Middle Temporal
    %  41 |    41 |   89/90 | Inferior Temporal gyrus
    aalG  = [1 2 13 3 14 4 5 6 7 8 15 9 10 11 16 17 18 19 20 21 12 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41];
    aal_L = [1:2:69 79:2:90];  % 41 left-hemisphere AAL indices
    aal_R = aal_L + 1;         % 41 right-hemisphere AAL indices

    for i = 1:41
        pidx_l(label_L.cdata == aalG(i)) = aal_L(i);
        pidx_r(label_R.cdata == aalG(i)) = aal_R(i);
    end
    return
else
    error("atlas " + atlasName + " not yet implemented")
end

for i = 1:numel(labels_l)
    pidx_l(label_L.cdata == labels_l(i)) = i;
end

for i = 1:numel(labels_r)
    pidx_r(label_R.cdata == labels_r(i)) = rh_extra_idx + i;
end


function v = parcel_to_vertex(parcelData, pidx, fillValue)
% Spread a parcel-wise vector onto vertices using the index from
% parcel_vertex_index. Vertices outside any parcel get fillValue.

v = repmat(fillValue, numel(pidx), 1);
inParcel = pidx > 0;
v(inParcel) = parcelData(pidx(inParcel));


function rasterize_panels(fig, dpi)
% Replace every axes of fig by an image of its OpenGL rendering, cropped
% from a capture of the whole figure at the given resolution. Images are
% trimmed to their content so exportgraphics still crops tightly.

axs = findobj(fig, 'Type', 'axes');
img = print(fig, '-RGBImage', '-opengl', sprintf('-r%d', dpi));
[H, W, ~] = size(img);
for ax = axs'
    pos = get(ax, 'Position');  % normalized units
    rows = max(1, round((1 - pos(2) - pos(4)) * H) + 1) : min(H, round((1 - pos(2)) * H));
    cols = max(1, round(pos(1) * W) + 1) : min(W, round((pos(1) + pos(3)) * W));
    delete(ax)

    panel = img(rows, cols, :);
    ink = any(panel ~= panel(1,1,:), 3);  % pixels differing from background
    if ~any(ink(:))
        continue
    end
    rows = rows(find(any(ink, 2), 1) : find(any(ink, 2), 1, 'last'));
    cols = cols(find(any(ink, 1), 1) : find(any(ink, 1), 1, 'last'));

    iax = axes(fig, 'Position', [(cols(1) - 1) / W, 1 - rows(end) / H, numel(cols) / W, numel(rows) / H]);
    image(iax, img(rows, cols, :))
    axis(iax, 'off')
end


function rgb = values_to_rgb(values, c, rangemin, rangemax)
% Map values into colormap c over [rangemin rangemax], as MATLAB does for
% scaled CData. Values outside the range are clipped to the end colors.

m = size(c, 1);
rangemin = double(rangemin);
rangemax = double(rangemax);
idx = fix((double(values(:)) - rangemin) / (rangemax - rangemin) * m) + 1;
idx(isnan(idx)) = 1;
idx = min(max(idx, 1), m);
rgb = c(idx, :);
