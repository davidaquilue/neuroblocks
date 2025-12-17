function render_patch(surf, cdata, cmin, cmax, viewang)

ax = gca;                    % current axes
axis(ax,'equal')             % no distortion
axis(ax,'off')               % hide axes

patch(ax, ...
    'Faces', surf.faces, ...
    'Vertices', surf.vertices, ...
    'FaceVertexCData', cdata, ...
    'FaceColor','interp', ...
    'EdgeColor','none');      % core rendering call

set(ax,'CLim',[cmin cmax])    % color limits
view(viewang)                 % camera angle
camlight                      % add light
lighting gouraud              % smooth lighting
material dull                 % reduce specular shine

end