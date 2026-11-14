%% NASA SPACE APPS - PLANETARY ANALOGUE AI DISCOVERY SYSTEM (FINAL PRO EDITION)
% Architettura: Unsupervised ML Data-Driven (PCA/SVD + K-Means + Dual-Metric Matching + Radar Chart)
% Input: 4 Cataloghi CSV agnostici (Luna, Terra-Luna, Marte, Terra-Marte)
% Layout: Cubo 3D PCA perfettamente centrato a sinistra + Radar Chart a destra + Mappe 2D sotto
clear; clc; close all;

fprintf('========================================================================================\n');
fprintf('     NASA SPACE APPS: PLANETARY ANALOGUE DISCOVERY SYSTEM (FINAL PRO EDITION)          \n');
fprintf('========================================================================================\n');

meta_cols = {'Site_Name', 'Country', 'Category', 'Latitude', 'Longitude', ...
             'Region_State', 'Geological_Setting', 'Best_Lunar_Match', ...
             'Primary_Use', 'Analog_Category', 'Operational_Status', 'Dominant_Mineral'};

%% ========================================================================================
%% PARTE I: MODULO LUNARE (LUNA <-> TERRA)
%% ========================================================================================

lunar_csv      = 'Lunar_Targets_Catalog.csv';
earth_luna_csv = 'Earth_Lunar_Analogs_Catalog.csv';

if ~isfile(lunar_csv) || ~isfile(earth_luna_csv)
    error('File CSV lunari mancanti! Assicurati che "%s" e "%s" siano nella cartella.', lunar_csv, earth_luna_csv);
end

T_moon  = readtable(lunar_csv);
T_earth = readtable(earth_luna_csv);

lunar_names   = T_moon.Site_Name;
coords_Moon   = [T_moon.Latitude, T_moon.Longitude];
earth_names   = T_earth.Site_Name;
earth_country = T_earth.Country;
coords_Earth  = [T_earth.Latitude, T_earth.Longitude];

lunar_feats = intersect(T_moon.Properties.VariableNames, T_earth.Properties.VariableNames, 'stable');
lunar_feats = setdiff(lunar_feats, meta_cols, 'stable');

X_Moon  = table2array(T_moon(:, lunar_feats));
X_Earth = table2array(T_earth(:, lunar_feats));
N_moon  = size(X_Moon, 1);
N_earth = size(X_Earth, 1);

fprintf('-> [LUNA] Caricati %d target, %d analoghi terrestri e %d feature quantitative.\n', ...
    N_moon, N_earth, length(lunar_feats));

%% Standardizzazione & PCA (Luna)
all_data = [X_Earth; X_Moon];
mu = mean(all_data, 1);
sigma = std(all_data, 0, 1); sigma(sigma == 0) = 1e-5;
all_data_norm = (all_data - mu) ./ sigma;

X_Earth_norm = all_data_norm(1:N_earth, :);
X_Moon_norm  = all_data_norm(N_earth+1:end, :);

[~, S, V] = svd(all_data_norm - mean(all_data_norm, 1), 'econ');
score_all = all_data_norm * V;

score_Earth_full = score_all(1:N_earth, :);
score_Moon_full  = score_all(N_earth+1:end, :);
score_Earth_3D   = score_all(1:N_earth, 1:3);
score_Moon_3D    = score_all(N_earth+1:end, 1:3);

eigVals = diag(S).^2 / (size(all_data_norm, 1) - 1);
explained = (eigVals / sum(eigVals)) * 100;

%% K-Means Clustering Terrestre (Luna)
k = 3; rng(42);
centroids = X_Earth_norm(randperm(N_earth, k), :);
cluster_idx = zeros(N_earth, 1);
for iter = 1:100
    dists = zeros(N_earth, k);
    for c = 1:k, dists(:, c) = sum((X_Earth_norm - centroids(c, :)).^2, 2); end
    [~, cluster_idx] = min(dists, [], 2);
    for c = 1:k
        pts = X_Earth_norm(cluster_idx == c, :);
        if ~isempty(pts), centroids(c, :) = mean(pts, 1); end
    end
end

%% Matching Bidirezionale (Luna)
moon_to_earth_best = zeros(N_moon, 1);
moon_to_earth_dist = zeros(N_moon, 1);
for m = 1:N_moon
    d = sqrt(sum((score_Earth_full - score_Moon_full(m, :)).^2, 2));
    [sorted_d, s_idx] = sort(d, 'ascend');
    moon_to_earth_best(m) = s_idx(1);
    moon_to_earth_dist(m) = sorted_d(1);
end

earth_to_moon_best = zeros(N_earth, 1);
earth_to_moon_dist = zeros(N_earth, 1);
for e = 1:N_earth
    d_rev = sqrt(sum((score_Moon_full - score_Earth_full(e, :)).^2, 2));
    [sorted_rev, s_rev_idx] = sort(d_rev, 'ascend');
    earth_to_moon_best(e) = s_rev_idx(1);
    earth_to_moon_dist(e) = sorted_rev(1);
end

%% Report Luna nella Command Window
fprintf('\n================== FLUSSO A: TARGET LUNARI -> ANALOGHI TERRESTRI ==================\n');
for m = 1:N_moon
    e_idx = moon_to_earth_best(m);
    fprintf('Target Luna: %s\n', lunar_names{m});
    fprintf('   -> Miglior Analogo: %s (%s) [Lat %6.2f, Lon %6.2f] | Distanza: %.3f\n\n', ...
        earth_names{e_idx}, earth_country{e_idx}, coords_Earth(e_idx,1), coords_Earth(e_idx,2), moon_to_earth_dist(m));
end

%% Mappa Luna
local_moon_file = 'moon_map.jpg';
if isfile(local_moon_file)
    moon_img = imread(local_moon_file);
else
    try
        moon_img = webread('https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/moon_1024.jpg', weboptions('Timeout', 10, 'CertificateFilename', ''));
        imwrite(moon_img, local_moon_file);
    catch
        [Xg, Yg] = meshgrid(linspace(-180, 180, 800), linspace(-90, 90, 400));
        m_s = 145 + 18*sin(Xg*pi/65).*cos(Yg*pi/35) + randn(400, 800)*5;
        moon_img = cat(3, uint8(m_s), uint8(m_s), uint8(m_s));
    end
end

%% CRUSCOTTO GRAFICO FIGURA 1 (LUNA + CUBO 3D CENTRATO + RADAR CHART)
fig1 = figure('Name', 'NASA Planetary Analogue Discovery System - LUNA', 'Color', 'w', 'WindowState', 'maximized');
sgtitle('NASA Planetary Analogue Dual-Planet Discovery System (LUNA <-> TERRA)', 'FontSize', 13, 'FontWeight', 'bold');

% Normalizzazione [0, 1] per il Radar Chart
min_v = min(all_data, [], 1); max_v = max(all_data, [], 1);
span_v = max_v - min_v; span_v(span_v == 0) = 1;
X_Moon_radar  = (X_Moon - min_v) ./ span_v;
X_Earth_radar = (X_Earth - min_v) ./ span_v;
clean_lunar_labels = strrep(lunar_feats, '_', ' ');

% 2. Radar Chart (In alto a destra, ben centrato sopra la Mappa Terrestre)
ax_radar1 = polaraxes('Parent', fig1, 'Position', [0.58, 0.51, 0.34, 0.32]);
e1_luna = moon_to_earth_best(1);
updateRadarPlot(ax_radar1, X_Moon_radar(1,:), X_Earth_radar(e1_luna,:), ...
    X_Moon(1,:), X_Earth(e1_luna,:), ...
    lunar_names{1}, earth_names{e1_luna}, moon_to_earth_dist(1), clean_lunar_labels, [0.95 0.75 0], 'r');

% 1. Spazio Latente 3D Cubico 1:1:1 (Perfettamente centrato nella meta sinistra!)
ax1 = subplot('Position', [0.13, 0.51, 0.28, 0.35]);
best_earth_unique = unique(moon_to_earth_best);

s_earth = scatter3(score_Earth_3D(:,1), score_Earth_3D(:,2), score_Earth_3D(:,3), 75, cluster_idx, 'filled', 'DisplayName', 'Siti Terrestri');
hold on;
s_moon = scatter3(score_Moon_3D(:,1), score_Moon_3D(:,2), score_Moon_3D(:,3), 160, 'y', 'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5, 'DisplayName', 'Target Lunari');
s_best = scatter3(score_Earth_3D(best_earth_unique, 1), score_Earth_3D(best_earth_unique, 2), score_Earth_3D(best_earth_unique, 3), 130, 'r', 'LineWidth', 2, 'DisplayName', 'Migliori Match');

max_lim_luna = ceil(max(abs([score_Earth_3D(:); score_Moon_3D(:)])));
xlim([-max_lim_luna max_lim_luna]); ylim([-max_lim_luna max_lim_luna]); zlim([-max_lim_luna max_lim_luna]);
pbaspect([1 1 1]); axis vis3d; box on;
xlabel(sprintf('PC1 (%.1f%%)', explained(1))); ylabel(sprintf('PC2 (%.1f%%)', explained(2))); zlabel(sprintf('PC3 (%.1f%%)', explained(3)));
title(sprintf('1. Spazio Latente PCA Cubico: %d Feature (Clicca un punto!)', length(lunar_feats)), 'FontSize', 10);
grid on; view(38, 24);

% Legenda indipendente per non schiacciare il cubo a sinistra
lgd1 = legend(ax1, 'show');
set(lgd1, 'Position', [0.41, 0.77, 0.10, 0.08], 'FontSize', 8);

% DataTip puliti per il Cubo 3D
moon_best_names = earth_names(moon_to_earth_best);
rev_lunar_names = lunar_names(earth_to_moon_best);
s_moon.DataTipTemplate.DataTipRows = [dataTipTextRow('Target Lunare', lunar_names), dataTipTextRow('Miglior Analogo Terra', moon_best_names), dataTipTextRow('Distanza PCA', moon_to_earth_dist, '%.3f')];
s_earth.DataTipTemplate.DataTipRows = [dataTipTextRow('Sito Terrestre', earth_names), dataTipTextRow('Nazione', earth_country), dataTipTextRow('Equivalente Lunare', rev_lunar_names)];
s_best.DataTipTemplate.DataTipRows = [dataTipTextRow('Miglior Match Terra', earth_names(best_earth_unique)), dataTipTextRow('Nazione', earth_country(best_earth_unique)), dataTipTextRow('Equivalente Lunare', rev_lunar_names(best_earth_unique))];

% 3. Mappa 2D della Luna (In basso a sinistra)
ax2 = subplot('Position', [0.05, 0.06, 0.43, 0.35]);
im_m = image([-180 180], [90 -90], moon_img);
set(im_m, 'PickableParts', 'none');
set(gca, 'YDir', 'normal'); hold on;
p_moon = scatter(coords_Moon(:, 2), coords_Moon(:, 1), 180, 'y', 'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.8, 'DisplayName', 'Target Lunari');
for m = 1:N_moon
    text(coords_Moon(m, 2) + 5, coords_Moon(m, 1) + 5, lunar_names{m}, ...
        'FontSize', 7.5, 'FontWeight', 'bold', 'Color', [1 1 0.2], 'BackgroundColor', [0 0 0 0.65], 'Margin', 1, 'PickableParts', 'none');
end
xlim([-180 180]); ylim([-90 90]); xticks(-180:60:180); yticks(-90:30:90);
grid on; set(gca, 'GridColor', [0.8 0.8 0.8], 'GridAlpha', 0.25);
xlabel('Longitudine Lunare (°)'); ylabel('Latitudine Lunare (°)');
title('3. Mappa Lunare Panoramica (Clicca una stella per aggiornare il Radar!)');
legend(p_moon, 'Location', 'southwest');
p_moon.DataTipTemplate.DataTipRows = [dataTipTextRow('Target Lunare', lunar_names), dataTipTextRow('Miglior Analogo Terra', moon_best_names), dataTipTextRow('Lat Lunare', coords_Moon(:,1)), dataTipTextRow('Lon Lunare', coords_Moon(:,2))];

% 4. Mappa 2D della Terra (In basso a destra)
ax3 = subplot('Position', [0.52, 0.06, 0.44, 0.35]);
m_all = geoscatter(coords_Earth(:,1), coords_Earth(:,2), 45, [0.35 0.35 0.35], 'filled', 'DisplayName', 'Siti Terrestri'); hold on;
m_best = geoscatter(coords_Earth(best_earth_unique, 1), coords_Earth(best_earth_unique, 2), 160, 'r', 'filled', 'LineWidth', 2.0, 'DisplayName', 'Migliori Match');
try geobasemap('colorterrain'); geolimits([-85 82], [-170 175]); catch, geobasemap('landcover'); end
for i = 1:length(best_earth_unique)
    idx_s = best_earth_unique(i);
    text(coords_Earth(idx_s, 1) + 2.2, coords_Earth(idx_s, 2) + 2.0, earth_names{idx_s}, ...
        'FontSize', 8, 'FontWeight', 'bold', 'Color', [0.75 0.1 0.1], 'PickableParts', 'none');
end
title('4. Mappa Topografica Terrestre (Clicca un sito per aggiornare il Radar!)'); legend('show', 'Location', 'southwest');
m_all.DataTipTemplate.DataTipRows = [dataTipTextRow('Sito Terrestre', earth_names), dataTipTextRow('Nazione', earth_country), dataTipTextRow('Equivalente Lunare', rev_lunar_names), dataTipTextRow('Latitudine', coords_Earth(:,1)), dataTipTextRow('Longitudine', coords_Earth(:,2))];
m_best.DataTipTemplate.DataTipRows = [dataTipTextRow('Miglior Match Terra', earth_names(best_earth_unique)), dataTipTextRow('Nazione', earth_country(best_earth_unique)), dataTipTextRow('Equivalente Lunare', rev_lunar_names(best_earth_unique)), dataTipTextRow('Latitudine', coords_Earth(best_earth_unique,1)), dataTipTextRow('Longitudine', coords_Earth(best_earth_unique,2))];

% Collegamento diretto al Click per aggiornare il Radar Chart (Zero puntini neri!)
set([s_moon, p_moon], 'ButtonDownFcn', @(src, evt) clickPlanetPoint(src, evt, ...
    moon_to_earth_best, moon_to_earth_dist, X_Moon_radar, X_Earth_radar, X_Moon, X_Earth, ...
    lunar_names, earth_names, clean_lunar_labels, ax_radar1, [0.95 0.75 0]));

set([s_earth, m_all], 'ButtonDownFcn', @(src, evt) clickEarthPoint(src, evt, ...
    1:N_earth, earth_to_moon_best, earth_to_moon_dist, X_Moon_radar, X_Earth_radar, X_Moon, X_Earth, ...
    lunar_names, earth_names, clean_lunar_labels, ax_radar1, [0.95 0.75 0]));

set([s_best, m_best], 'ButtonDownFcn', @(src, evt) clickEarthPoint(src, evt, ...
    best_earth_unique, earth_to_moon_best, earth_to_moon_dist, X_Moon_radar, X_Earth_radar, X_Moon, X_Earth, ...
    lunar_names, earth_names, clean_lunar_labels, ax_radar1, [0.95 0.75 0]));

drawnow; pause(0.5);


%% ========================================================================================
%% PARTE II: MODULO MARZIANO (MARTE <-> TERRA)
%% ========================================================================================

fprintf('\n========================================================================================\n');
fprintf('     NASA SPACE APPS: PLANETARY ANALOGUE DISCOVERY SYSTEM (MARS MODULE)                \n');
fprintf('========================================================================================\n');

mars_csv       = 'Mars_Targets_Catalog.csv';
earth_mars_csv = 'Earth_Mars_Analogs_Catalog.csv';

if ~isfile(mars_csv) || ~isfile(earth_mars_csv)
    error('File CSV marziani mancanti! Assicurati che "%s" e "%s" siano nella cartella.', mars_csv, earth_mars_csv);
end

T_mars       = readtable(mars_csv);
T_mars_earth = readtable(earth_mars_csv);

mars_names  = T_mars.Site_Name;
coords_Mars = [T_mars.Latitude, T_mars.Longitude];

mars_earth_names    = T_mars_earth.Site_Name;
mars_earth_country  = T_mars_earth.Country;
mars_earth_category = T_mars_earth.Category;
coords_Mars_Earth   = [T_mars_earth.Latitude, T_mars_earth.Longitude];

mars_feats = intersect(T_mars.Properties.VariableNames, T_mars_earth.Properties.VariableNames, 'stable');
mars_feats = setdiff(mars_feats, meta_cols, 'stable');

X_Mars       = table2array(T_mars(:, mars_feats));
X_Mars_Earth = table2array(T_mars_earth(:, mars_feats));
N_mars       = size(X_Mars, 1);
N_mars_earth = size(X_Mars_Earth, 1);

fprintf('-> [MARTE] Caricati %d target, %d analoghi terrestri e %d feature quantitative.\n', ...
    N_mars, N_mars_earth, length(mars_feats));

%% Domain-Adapted Z-Score & PCA (Marte)
mu_E_m = mean(X_Mars_Earth, 1); sd_E_m = std(X_Mars_Earth, 0, 1); sd_E_m(sd_E_m == 0) = 1e-5;
X_Mars_Earth_norm = (X_Mars_Earth - mu_E_m) ./ sd_E_m;

mu_M_m = mean(X_Mars, 1); sd_M_m = std(X_Mars, 0, 1); sd_M_m(sd_M_m == 0) = 1e-5;
X_Mars_norm = (X_Mars - mu_M_m) ./ sd_M_m;

all_mars_data_norm = [X_Mars_Earth_norm; X_Mars_norm];

[~, S_mars, V_mars] = svd(all_mars_data_norm - mean(all_mars_data_norm, 1), 'econ');
score_all_mars = all_mars_data_norm * V_mars;

score_Mars_Earth_full = score_all_mars(1:N_mars_earth, :);
score_Mars_full       = score_all_mars(N_mars_earth+1:end, :);
score_Mars_Earth_3D   = score_all_mars(1:N_mars_earth, 1:3);
score_Mars_3D         = score_all_mars(N_mars_earth+1:end, 1:3);

eigVals_mars = diag(S_mars).^2 / (size(all_mars_data_norm, 1) - 1);
explained_mars = (eigVals_mars / sum(eigVals_mars)) * 100;

%% K-Means Clustering Terrestre (Marte)
k_mars = 3; rng(42);
centroids_mars = X_Mars_Earth_norm(randperm(N_mars_earth, k_mars), :);
cluster_idx_mars = zeros(N_mars_earth, 1);
for iter = 1:100
    dists_m = zeros(N_mars_earth, k_mars);
    for c = 1:k_mars, dists_m(:, c) = sum((X_Mars_Earth_norm - centroids_mars(c, :)).^2, 2); end
    [~, cluster_idx_mars] = min(dists_m, [], 2);
    for c = 1:k_mars
        pts_m = X_Mars_Earth_norm(cluster_idx_mars == c, :);
        if ~isempty(pts_m), centroids_mars(c, :) = mean(pts_m, 1); end
    end
end

%% Matching Bidirezionale (Marte)
mars_to_earth_best = zeros(N_mars, 1);
mars_to_earth_dist = zeros(N_mars, 1);
mars_to_earth_cos  = zeros(N_mars, 1);
for m = 1:N_mars
    d_m = sqrt(sum((score_Mars_Earth_full - score_Mars_full(m, :)).^2, 2));
    [sorted_dm, sm_idx] = sort(d_m, 'ascend');
    best_e = sm_idx(1);
    mars_to_earth_best(m) = best_e;
    mars_to_earth_dist(m) = sorted_dm(1);
    mars_to_earth_cos(m)  = dot(score_Mars_Earth_full(best_e, :), score_Mars_full(m, :)) / ...
                            (norm(score_Mars_Earth_full(best_e, :)) * norm(score_Mars_full(m, :)));
end

earth_to_mars_best = zeros(N_mars_earth, 1);
earth_to_mars_dist = zeros(N_mars_earth, 1);
for e = 1:N_mars_earth
    d_em = sqrt(sum((score_Mars_full - score_Mars_Earth_full(e, :)).^2, 2));
    [sorted_dem, sem_idx] = sort(d_em, 'ascend');
    earth_to_mars_best(e) = sem_idx(1);
    earth_to_mars_dist(e) = sorted_dem(1);
end

%% Report Marte nella Command Window
fprintf('\n================== FLUSSO C: TARGET MARZIANI -> ANALOGHI TERRESTRI ==================\n');
for m = 1:N_mars
    e_idx = mars_to_earth_best(m);
    fprintf('Target Marte: %s\n', mars_names{m});
    fprintf('   -> Miglior Analogo: %s (%s) | Distanza: %.3f | Cosine Sim: %.4f\n\n', ...
        mars_earth_names{e_idx}, mars_earth_country{e_idx}, mars_to_earth_dist(m), mars_to_earth_cos(m));
end

%% Mappa Marte
local_mars_file = 'mars_map.jpg';
if isfile(local_mars_file)
    mars_img = imread(local_mars_file);
else
    try
        mars_img = webread('https://upload.wikimedia.org/wikipedia/commons/thumb/0/02/OSIRIS_Mars_true_color.jpg/1280px-OSIRIS_Mars_true_color.jpg', weboptions('Timeout', 15, 'CertificateFilename', ''));
        imwrite(mars_img, local_mars_file);
    catch
        [Xg_m, Yg_m] = meshgrid(linspace(-180, 180, 1000), linspace(-90, 90, 500));
        m_red = 175 + 25*sin(Xg_m*pi/80).*cos(Yg_m*pi/40) + randn(500, 1000)*4;
        mars_img = cat(3, uint8(m_red), uint8(double(m_red)*0.52), uint8(double(m_red)*0.32));
    end
end

%% CRUSCOTTO GRAFICO FIGURA 2 (MARTE + CUBO 3D CENTRATO + RADAR CHART)
fig2 = figure('Name', 'NASA Planetary Analogue Discovery System - MARTE', 'Color', 'w', 'WindowState', 'maximized');
sgtitle('NASA Planetary Analogue Dual-Planet Discovery System (MARTE <-> TERRA)', 'FontSize', 13, 'FontWeight', 'bold');

% Normalizzazione [0, 1] per il Radar Chart di Marte
min_em = min(X_Mars_Earth, [], 1); max_em = max(X_Mars_Earth, [], 1);
span_em = max_em - min_em; span_em(span_em == 0) = 1;
X_Earth_Mars_radar = (X_Mars_Earth - min_em) ./ span_em;

min_mm = min(X_Mars, [], 1); max_mm = max(X_Mars, [], 1);
span_mm = max_mm - min_mm; span_mm(span_mm == 0) = 1;
X_Mars_radar = (X_Mars - min_mm) ./ span_mm;
clean_mars_labels = strrep(mars_feats, '_', ' ');

% 2. Radar Chart (In alto a destra, ben centrato sopra la Mappa Terrestre)
ax_radar2 = polaraxes('Parent', fig2, 'Position', [0.58, 0.51, 0.34, 0.32]);
e1_mars = mars_to_earth_best(1);
updateRadarPlot(ax_radar2, X_Mars_radar(1,:), X_Earth_Mars_radar(e1_mars,:), ...
    X_Mars(1,:), X_Mars_Earth(e1_mars,:), ...
    mars_names{1}, mars_earth_names{e1_mars}, mars_to_earth_dist(1), clean_mars_labels, [1 0.45 0], 'r');

% 1. Spazio Latente 3D Cubico 1:1:1 (Perfettamente centrato nella meta sinistra!)
ax1_m = subplot('Position', [0.13, 0.51, 0.28, 0.35]);
best_mars_earth_unique = unique(mars_to_earth_best);

s_earth_m = scatter3(score_Mars_Earth_3D(:,1), score_Mars_Earth_3D(:,2), score_Mars_Earth_3D(:,3), 80, cluster_idx_mars, 'filled', 'DisplayName', 'Siti Terrestri');
hold on;
s_mars_pts = scatter3(score_Mars_3D(:,1), score_Mars_3D(:,2), score_Mars_3D(:,3), 170, [1 0.35 0.1], 'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.8, 'DisplayName', 'Target Marziani');
s_best_m = scatter3(score_Mars_Earth_3D(best_mars_earth_unique, 1), score_Mars_Earth_3D(best_mars_earth_unique, 2), score_Mars_Earth_3D(best_mars_earth_unique, 3), 135, 'r', 'LineWidth', 2.2, 'DisplayName', 'Migliori Match');

max_lim_mars = ceil(max(abs([score_Mars_Earth_3D(:); score_Mars_3D(:)])));
xlim([-max_lim_mars max_lim_mars]); ylim([-max_lim_mars max_lim_mars]); zlim([-max_lim_mars max_lim_mars]);
pbaspect([1 1 1]); axis vis3d; box on;
xlabel(sprintf('PC1 (%.1f%%)', explained_mars(1))); ylabel(sprintf('PC2 (%.1f%%)', explained_mars(2))); zlabel(sprintf('PC3 (%.1f%%)', explained_mars(3)));
title(sprintf('1. Spazio Latente PCA Cubico: %d Feature (Clicca un punto!)', length(mars_feats)), 'FontSize', 10);
grid on; view(38, 24);

% Legenda indipendente per non schiacciare il cubo a sinistra
lgd2 = legend(ax1_m, 'show');
set(lgd2, 'Position', [0.41, 0.77, 0.10, 0.08], 'FontSize', 8);

% DataTip puliti per il Cubo 3D di Marte
mars_best_names = mars_earth_names(mars_to_earth_best);
rev_mars_names  = mars_names(earth_to_mars_best);
s_mars_pts.DataTipTemplate.DataTipRows = [dataTipTextRow('Target Marziano', mars_names), dataTipTextRow('Miglior Analogo Terra', mars_best_names), dataTipTextRow('Distanza PCA', mars_to_earth_dist, '%.3f')];
s_earth_m.DataTipTemplate.DataTipRows = [dataTipTextRow('Sito Terrestre', mars_earth_names), dataTipTextRow('Nazione', mars_earth_country), dataTipTextRow('Categoria', mars_earth_category), dataTipTextRow('Equivalente Marziano', rev_mars_names)];
s_best_m.DataTipTemplate.DataTipRows = [dataTipTextRow('Miglior Match Terra', mars_earth_names(best_mars_earth_unique)), dataTipTextRow('Nazione', mars_earth_country(best_mars_earth_unique)), dataTipTextRow('Equivalente Marziano', rev_mars_names(best_mars_earth_unique))];

% 3. Mappa 2D di Marte (In basso a sinistra)
ax2_m = subplot('Position', [0.05, 0.06, 0.43, 0.35]);
im_mars = image([-180 180], [90 -90], mars_img);
set(im_mars, 'PickableParts', 'none');
set(gca, 'YDir', 'normal'); hold on;
p_mars = scatter(coords_Mars(:, 2), coords_Mars(:, 1), 180, [1 0.85 0.1], 'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.8, 'DisplayName', 'Target Marziani');
for m = 1:N_mars
    text(coords_Mars(m, 2) + 5, coords_Mars(m, 1) + 5, mars_names{m}, ...
        'FontSize', 7, 'FontWeight', 'bold', 'Color', [1 0.95 0.3], 'BackgroundColor', [0 0 0 0.65], 'Margin', 1, 'PickableParts', 'none');
end
xlim([-180 180]); ylim([-90 90]); xticks(-180:60:180); yticks(-90:30:90);
grid on; set(gca, 'GridColor', [0.8 0.8 0.8], 'GridAlpha', 0.25);
xlabel('Longitudine Marziana (°)'); ylabel('Latitudine Marziana (°)');
title('3. Mappa Marziana Panoramica (Clicca una stella per aggiornare il Radar!)');
legend(p_mars, 'Location', 'southwest');
p_mars.DataTipTemplate.DataTipRows = [dataTipTextRow('Target Marziano', mars_names), dataTipTextRow('Miglior Analogo Terra', mars_best_names), dataTipTextRow('Lat Marziana', coords_Mars(:,1)), dataTipTextRow('Lon Marziana', coords_Mars(:,2))];

% 4. Mappa 2D della Terra (In basso a destra)
ax3_m = subplot('Position', [0.52, 0.06, 0.44, 0.35]);
m_all_m = geoscatter(coords_Mars_Earth(:,1), coords_Mars_Earth(:,2), 45, [0.25 0.25 0.25], 'filled', 'DisplayName', 'Siti Terrestri'); hold on;
m_best_m = geoscatter(coords_Mars_Earth(best_mars_earth_unique, 1), coords_Mars_Earth(best_mars_earth_unique, 2), 160, 'r', 'filled', 'LineWidth', 2.0, 'DisplayName', 'Migliori Match');
try geobasemap('colorterrain'); geolimits([-55 82], [-145 155]); catch, geobasemap('landcover'); end
for i = 1:length(best_mars_earth_unique)
    idx_s = best_mars_earth_unique(i);
    text(coords_Mars_Earth(idx_s, 1) + 2.5, coords_Mars_Earth(idx_s, 2) + 2.0, mars_earth_names{idx_s}, ...
        'FontSize', 8, 'FontWeight', 'bold', 'Color', [0.75 0.05 0.05], 'PickableParts', 'none');
end
title('4. Mappa Topografica Terrestre (Clicca un sito per aggiornare il Radar!)'); legend('show', 'Location', 'southwest');
m_all_m.DataTipTemplate.DataTipRows = [dataTipTextRow('Sito Terrestre', mars_earth_names), dataTipTextRow('Nazione', mars_earth_country), dataTipTextRow('Categoria', mars_earth_category), dataTipTextRow('Equivalente Marziano', rev_mars_names), dataTipTextRow('Latitudine', coords_Mars_Earth(:,1)), dataTipTextRow('Longitudine', coords_Mars_Earth(:,2))];
m_best_m.DataTipTemplate.DataTipRows = [dataTipTextRow('Miglior Match Terra', mars_earth_names(best_mars_earth_unique)), dataTipTextRow('Nazione', mars_earth_country(best_mars_earth_unique)), dataTipTextRow('Categoria', mars_earth_category(best_mars_earth_unique)), dataTipTextRow('Equivalente Marziano', rev_mars_names(best_mars_earth_unique)), dataTipTextRow('Latitudine', coords_Mars_Earth(best_mars_earth_unique,1)), dataTipTextRow('Longitudine', coords_Mars_Earth(best_mars_earth_unique,2))];

% Collegamento diretto al Click per aggiornare il Radar Chart (Zero puntini neri!)
set([s_mars_pts, p_mars], 'ButtonDownFcn', @(src, evt) clickPlanetPoint(src, evt, ...
    mars_to_earth_best, mars_to_earth_dist, X_Mars_radar, X_Earth_Mars_radar, X_Mars, X_Mars_Earth, ...
    mars_names, mars_earth_names, clean_mars_labels, ax_radar2, [1 0.45 0]));

set([s_earth_m, m_all_m], 'ButtonDownFcn', @(src, evt) clickEarthPoint(src, evt, ...
    1:N_mars_earth, earth_to_mars_best, earth_to_mars_dist, X_Mars_radar, X_Earth_Mars_radar, X_Mars, X_Mars_Earth, ...
    mars_names, mars_earth_names, clean_mars_labels, ax_radar2, [1 0.45 0]));

set([s_best_m, m_best_m], 'ButtonDownFcn', @(src, evt) clickEarthPoint(src, evt, ...
    best_mars_earth_unique, earth_to_mars_best, earth_to_mars_dist, X_Mars_radar, X_Earth_Mars_radar, X_Mars, X_Mars_Earth, ...
    mars_names, mars_earth_names, clean_mars_labels, ax_radar2, [1 0.45 0]));

drawnow;


%% ========================================================================================
%% FUNZIONI AUSILIARIE PER IL RADAR CHART CON TOOLTIP DELLE FEATURE
%% ========================================================================================

function s_short = shortSiteName(s_full)
    idx_p = strfind(s_full, '(');
    if ~isempty(idx_p)
        s_short = strtrim(s_full(1:idx_p(1)-1));
    else
        s_short = s_full;
    end
end

function updateRadarPlot(pax, vec_planet, vec_earth, raw_planet, raw_earth, name_planet, name_earth, dist_val, feat_labels, col_planet, col_earth)
    cla(pax);
    num_f = length(vec_planet);
    angles_deg = linspace(0, 360, num_f + 1);
    angles_rad = deg2rad(angles_deg);
    
    r_p_closed   = [vec_planet(:)', vec_planet(1)];
    r_e_closed   = [vec_earth(:)',  vec_earth(1)];
    raw_p_closed = [raw_planet(:)', raw_planet(1)];
    raw_e_closed = [raw_earth(:)',  raw_earth(1)];
    lbl_closed   = [feat_labels(:)', feat_labels(1)];
    
    p_planet_line = polarplot(pax, angles_rad, r_p_closed, '-o', 'LineWidth', 2.2, ...
        'Color', col_planet, 'MarkerFaceColor', col_planet, 'MarkerSize', 5, 'DisplayName', name_planet);
    hold(pax, 'on');
    p_earth_line = polarplot(pax, angles_rad, r_e_closed, '-s', 'LineWidth', 2.0, ...
        'Color', col_earth, 'MarkerFaceColor', col_earth, 'MarkerSize', 5, 'DisplayName', name_earth);
    hold(pax, 'off');
    
    site_p_rep = repmat({name_planet}, 1, num_f + 1);
    site_e_rep = repmat({name_earth},  1, num_f + 1);
    
    p_planet_line.DataTipTemplate.DataTipRows = [ ...
        dataTipTextRow('Sito Planetario', site_p_rep), ...
        dataTipTextRow('Feature', lbl_closed), ...
        dataTipTextRow('Valore Reale (SI)', raw_p_closed, '%.2f'), ...
        dataTipTextRow('Livello Normalizzato (%)', r_p_closed * 100, '%.1f%%')];
        
    p_earth_line.DataTipTemplate.DataTipRows = [ ...
        dataTipTextRow('Sito Terrestre', site_e_rep), ...
        dataTipTextRow('Feature', lbl_closed), ...
        dataTipTextRow('Valore Reale (SI)', raw_e_closed, '%.2f'), ...
        dataTipTextRow('Livello Normalizzato (%)', r_e_closed * 100, '%.1f%%')];
    
    pax.ThetaTick = angles_deg(1:end-1);
    pax.ThetaTickLabel = feat_labels;
    pax.FontSize = 7.5;
    rlim(pax, [0 1]);
    rticks(pax, [0.25 0.5 0.75 1.0]);
    rticklabels(pax, {'25%', '50%', '75%', '100%'});
    
    short_p = shortSiteName(name_planet);
    short_e = shortSiteName(name_earth);
    title(pax, {'2. Radar Chart: Firma Geofisica Comparata', ...
        sprintf('%s  vs  %s (Dist. PCA: %.2f)', short_p, short_e, dist_val)}, ...
        'FontSize', 9, 'FontWeight', 'bold');
    legend(pax, 'Location', 'southoutside', 'FontSize', 7.5);
end

function idx = findClickedIndex(src, evt)
    pt = evt.IntersectionPoint;
    if isprop(src, 'LatitudeData')
        lats = src.LatitudeData(:);
        lons = src.LongitudeData(:);
        [~, idx] = min((lats - pt(1)).^2 + (lons - pt(2)).^2);
    else
        xd = src.XData(:);
        yd = src.YData(:);
        if isprop(src, 'ZData') && ~isempty(src.ZData) && length(pt) >= 3
            zd = src.ZData(:);
            [~, idx] = min((xd - pt(1)).^2 + (yd - pt(2)).^2 + (zd - pt(3)).^2);
        else
            [~, idx] = min((xd - pt(1)).^2 + (yd - pt(2)).^2);
        end
    end
end

function clickPlanetPoint(src, evt, p2e_best, p2e_dist, X_P_rad, X_E_rad, X_P_raw, X_E_raw, p_names, e_names, feat_labels, pax, col_planet)
    p_idx = findClickedIndex(src, evt);
    e_idx = p2e_best(p_idx);
    dist_val = p2e_dist(p_idx);
    updateRadarPlot(pax, X_P_rad(p_idx, :), X_E_rad(e_idx, :), ...
        X_P_raw(p_idx, :), X_E_raw(e_idx, :), ...
        p_names{p_idx}, e_names{e_idx}, dist_val, feat_labels, col_planet, 'r');
end

function clickEarthPoint(src, evt, idx_map, e2p_best, e2p_dist, X_P_rad, X_E_rad, X_P_raw, X_E_raw, p_names, e_names, feat_labels, pax, col_planet)
    raw_idx = findClickedIndex(src, evt);
    e_idx = idx_map(raw_idx);
    p_idx = e2p_best(e_idx);
    dist_val = e2p_dist(e_idx);
    updateRadarPlot(pax, X_P_rad(p_idx, :), X_E_rad(e_idx, :), ...
        X_P_raw(p_idx, :), X_E_raw(e_idx, :), ...
        p_names{p_idx}, e_names{e_idx}, dist_val, feat_labels, col_planet, 'r');
end
