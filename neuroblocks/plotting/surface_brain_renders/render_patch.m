function render_patch(surf, cdata, cmin, cmax, viewang, outline, faceColor)
% RENDER_PATCH draws a cortical surface in the current axes.
%
% surf      : struct/gifti with faces and vertices
% cdata     : per-vertex data, Nx1 (scaled through colormap) or Nx3 truecolor
% cmin, cmax: color limits of the axes
% viewang   : camera angle passed to view
%
% Optional:
% outline   : struct with fields
%               indicator : Nx1 per-vertex 0/1, contour drawn at level 0.5
%               color     : 1x3 RGB line color
%               width     : line width
%             Empty or missing draws no outline.
% faceColor : 'interp' (default) or 'flat'

if ~exist('outline','var')
    outline = [];
end

if ~exist('faceColor','var') || isempty(faceColor)
    faceColor = 'interp';
end

ax = gca;                    % current axes
axis(ax,'equal')             % no distortion
axis(ax,'off')               % hide axes

p = patch(ax, ...
    'Faces', surf.faces, ...
    'Vertices', surf.vertices, ...
    'FaceVertexCData', cdata, ...
    'FaceColor', faceColor, ...
    'EdgeColor','none');      % core rendering call

set(ax,'CLim',[cmin cmax])    % color limits
view(viewang)                 % camera angle
camlight                      % add light
lighting gouraud              % smooth lighting
material dull                 % reduce specular shine

if ~isempty(outline) && any(outline.indicator(:))
    draw_outline(ax, p, outline)
end

end


function draw_outline(ax, p, outline)
% Contour at level 0.5 of a per-vertex 0/1 indicator. Every face whose
% vertices are not all equal has exactly two edges with different end
% values; a segment joins the midpoints of those two edges.

offset = 0.4;  % mm, lifts the line off the surface so it is not covered

F = double(p.Faces);
V = double(p.Vertices);
val = double(outline.indicator(:) ~= 0);
val = val(F);

d12 = val(:,1) ~= val(:,2);
d23 = val(:,2) ~= val(:,3);
d31 = val(:,3) ~= val(:,1);
crossed = d12 | d23;
if ~any(crossed)
    return
end
F = F(crossed,:);
d12 = d12(crossed);
d31 = d31(crossed);

% Vertex normals of the patch, flipped to face the camera (patch normals
% follow face winding, which points inward on these surfaces; flat maps
% have no inside). Hidden back faces are covered by the surface anyway.
drawnow
N = double(p.VertexNormals);
N = N ./ max(vecnorm(N, 2, 2), eps);
camdir = campos(ax) - camtarget(ax);
camdir = camdir / norm(camdir);
s = sign(N * camdir');
s(s == 0) = 1;
W = V + offset * (N .* s);

m12 = (W(F(:,1),:) + W(F(:,2),:)) / 2;
m23 = (W(F(:,2),:) + W(F(:,3),:)) / 2;
m31 = (W(F(:,3),:) + W(F(:,1),:)) / 2;

% Two of d12, d23, d31 are true: A is on edge 12 or 23, B on edge 31 or 23
A = m23; A(d12,:) = m12(d12,:);
B = m23; B(d31,:) = m31(d31,:);

nanCol = nan(size(A,1), 1);
X = [A(:,1) B(:,1) nanCol]';
Y = [A(:,2) B(:,2) nanCol]';
Z = [A(:,3) B(:,3) nanCol]';

line(ax, X(:), Y(:), Z(:), ...
    'Color', outline.color, ...
    'LineWidth', outline.width);

end
