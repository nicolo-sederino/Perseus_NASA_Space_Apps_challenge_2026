%% NASA SPACE APPS - PLANETARY ANALOGUE AI DISCOVERY SYSTEM (FINAL PRO EDITION)
% Architettura: Unsupervised ML Data-Driven (PCA/SVD + K-Means + Dual-Metric Matching + Radar Chart + Multi-Test Finder)
% Input: 4 Cataloghi CSV agnostici (Luna, Terra-Luna, Marte, Terra-Marte)
% Layout: Cubo 3D PCA a sinistra + Radar Chart a destra + Mappe 2D sotto + Ricerca Siti Multi-Test

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

% Carica i dati
T_moon  = readtable(lunar_csv);
T_earth = readtable(earth_luna_csv);

lunar_names   = T_moon.Site_Name;
coords_Moon   = [T_moon.Latitude, T_moon.Longitude];
earth_names   = T_earth.Site_Name;
earth_country = T_earth.Country;
coords_Earth  = [T_earth.Latitude, T_earth.Longitude];

% Estrae automaticamente la colonna delle categorie per i siti terrestri lunari
if ismember('Category', T_earth.Properties.VariableNames)
    earth_luna_category = T_earth.Category;
elseif ismember('Analog_Category', T_earth.Properties.VariableNames)
    earth_luna_category = T_earth.Analog_Category;
elseif ismember('Geological_Setting', T_earth.Properties.VariableNames)
    earth_luna_category = T_earth.Geological_Setting;
else
    earth_luna_category = repmat({'Lunar Analog'}, height(T_earth), 1);
end

% Confronta le colonne dei due file ed estrae solo le feature numeriche quantitative
lunar_feats = intersect(T_moon.Properties.VariableNames, T_earth.Properties.VariableNames, 'stable');
lunar_feats = setdiff(lunar_feats, meta_cols, 'stable');

X_Moon  = table2array(T_moon(:, lunar_feats));
X_Earth = table2array(T_earth(:, lunar_feats));
N_moon  = size(X_Moon, 1);
N_earth = size(X_Earth, 1);

fprintf('-> [LUNA] Caricati %d target, %d analoghi terrestri e %d features quantitative.\n', ...
    N_moon, N_earth, length(lunar_feats));

% Standardizzazione & PCA (Luna)
all_data = [X_Earth; X_Moon];
mu = mean(all_data, 1);
sigma = std(all_data, 0, 1);
sigma(sigma == 0) = 1e-5;
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

% K-Means Clustering Terrestre (Luna)
k = 5;
rng(42);
centroids = X_Earth_norm(randperm(N_earth, k), :);
cluster_idx = zeros(N_earth, 1);
for iter = 1:100
    dists = zeros(N_earth, k);
    for c = 1:k
        dists(:, c) = sum((X_Earth_norm - centroids(c, :)).^2, 2);
    end
    [~, cluster_idx] = min(dists, [], 2);
    for c = 1:k
        pts = X_Earth_norm(cluster_idx == c, :);
        if ~isempty(pts)
            centroids(c, :) = mean(pts, 1);
        end
    end
end

% Matching Bidirezionale (Luna)
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

% Report Luna nella Command Window (Flusso A + Flusso B Inverso)
fprintf('\n================== FLUSSO A: TARGET LUNARI -> ANALOGHI TERRESTRI ==================\n');
for m = 1:N_moon
    e_idx = moon_to_earth_best(m);
    fprintf('Target Luna: %s\n', lunar_names{m});
    fprintf('   -> Miglior Analogo: %s (%s) [Lat %6.2f, Lon %6.2f] | Distanza: %.3f\n\n', ...
        earth_names{e_idx}, earth_country{e_idx}, coords_Earth(e_idx,1), coords_Earth(e_idx,2), moon_to_earth_dist(m));
end

fprintf('\n================== FLUSSO B: ANALOGHI TERRESTRI -> TARGET LUNARI ==================\n');
for e = 1:N_earth
    m_idx = earth_to_moon_best(e);
    fprintf('Sito Terrestre: %s (%s) [%s]\n', earth_names{e}, earth_country{e}, earth_luna_category{e});
    fprintf('   -> Equivalente Lunare: %s | Distanza PCA: %.3f\n\n', ...
        lunar_names{m_idx}, earth_to_moon_dist(e));
end

% Mappa Luna
local_moon_file = 'moon_map.jpg';
if isfile(local_moon_file)
    moon_img = imread(local_moon_file);
else
    try
        moon_img = webread('https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/moon_1024.jpg', weboptions('Timeout', 10, 'CertificateFilename', ''));
        imwrite(moon_img, local_moon_file);
    catch
        [Xg, Yg] = meshgrid(linspace(-180, 180, 800), linspace(-90, 90, 400));
        m_s = 145 + 18 * sin(Xg * pi/65) .* cos(Yg * pi/35) + randn(400, 800)*5;
        moon_img = cat(3, uint8(m_s), uint8(m_s), uint8(m_s));
    end
end

%% CRUSCOTTO GRAFICO FIGURA 1 (LUNA + CUBO 3D CENTRATO + RADAR CHART)
fig1 = figure('Name', 'NASA Planetary Analogue Discovery System - LUNA', 'Color', 'w', 'WindowState', 'maximized');
sgtitle('NASA Planetary Analogue Dual-Planet Discovery System (LUNA <-> TERRA)', 'FontSize', 13, 'FontWeight', 'bold');

% Normalizzazione [0, 1] per il Radar Chart
min_v = min(all_data, [], 1);
max_v = max(all_data, [], 1);
span_v = max_v - min_v;
span_v(span_v == 0) = 1;
X_Moon_radar  = (X_Moon - min_v) ./ span_v;
X_Earth_radar = (X_Earth - min_v) ./ span_v;
clean_lunar_labels = strrep(lunar_feats, '_', ' ');

% 2. Radar Chart (In alto a destra)
ax_radar1 = polaraxes('Parent', fig1, 'Position', [0.58, 0.51, 0.34, 0.32]);
e1_luna = moon_to_earth_best(1);
updateRadarPlot(ax_radar1, X_Moon_radar(1,:), X_Earth_radar(e1_luna,:), ...
    X_Moon(1,:), X_Earth(e1_luna,:), ...
    lunar_names{1}, earth_names{e1_luna}, moon_to_earth_dist(1), clean_lunar_labels, [0.95 0.75 0], 'r');

% 1. Spazio Latente 3D Cubico (In alto a sinistra)
ax1 = subplot('Position', [0.10, 0.54, 0.19, 0.24]);
hold(ax1, 'on');
best_earth_unique = unique(moon_to_earth_best);
moon_best_names = earth_names(moon_to_earth_best);
rev_lunar_names = lunar_names(earth_to_moon_best);

colori_cluster_luna = parula(k);
s_earth = gobjects(k, 1);

for c = 1:k
    idx_c = find(cluster_idx == c);
    if isempty(idx_c), continue; end
    
    cats_c = strtrim(regexprep(earth_luna_category(idx_c), '\s*\(.*?\)', ''));
    [u_cats, ~, ic_cats] = unique(cats_c);
    counts = accumarray(ic_cats, 1);
    [~, sort_order] = sort(counts, 'descend');
    u_cats_sorted = u_cats(sort_order);
    
    if isscalar(u_cats_sorted)
        lbl_c = sprintf('C%d: %s', c, u_cats_sorted{1});
    else
        lbl_c = sprintf('C%d: %s / %s', c, u_cats_sorted{1}, u_cats_sorted{2});
    end
    
    s_earth(c) = scatter3(ax1, ...
        score_Earth_3D(idx_c, 1), score_Earth_3D(idx_c, 2), score_Earth_3D(idx_c, 3), ...
        45, colori_cluster_luna(c, :), 'filled', 'MarkerEdgeColor', [0.2 0.2 0.2], 'LineWidth', 0.4, 'DisplayName', lbl_c);
    
    cluster_tag = repmat({lbl_c}, length(idx_c), 1);
    s_earth(c).DataTipTemplate.DataTipRows = [ ...
        dataTipTextRow('Sito Terrestre', earth_names(idx_c)), ...
        dataTipTextRow('Nazione', earth_country(idx_c)), ...
        dataTipTextRow('Cluster K-Means', cluster_tag), ...
        dataTipTextRow('Categoria CSV', earth_luna_category(idx_c)), ...
        dataTipTextRow('Equivalente Lunare', rev_lunar_names(idx_c))];
        
    set(s_earth(c), 'ButtonDownFcn', @(src, evt) clickEarthPoint(src, evt, ...
        idx_c, earth_to_moon_best, earth_to_moon_dist, X_Moon_radar, X_Earth_radar, X_Moon, X_Earth, ...
        lunar_names, earth_names, clean_lunar_labels, ax_radar1, [0.95 0.75 0]));
end

s_moon = scatter3(ax1, score_Moon_3D(:,1), score_Moon_3D(:,2), score_Moon_3D(:,3), ...
    95, 'y', 'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.2, 'DisplayName', 'Target Lunari');
s_best = scatter3(ax1, score_Earth_3D(best_earth_unique, 1), score_Earth_3D(best_earth_unique, 2), score_Earth_3D(best_earth_unique, 3), ...
    85, 'r', 'LineWidth', 1.6, 'DisplayName', 'Migliori Match');

all_pts_luna = [score_Earth_3D; score_Moon_3D];
min_l = min(all_pts_luna, [], 1);
max_l = max(all_pts_luna, [], 1);
pad_l = (max_l - min_l) * 0.10;
xlim(ax1, [min_l(1)-pad_l(1), max_l(1)+pad_l(1)]);
ylim(ax1, [min_l(2)-pad_l(2), max_l(2)+pad_l(2)]);
zlim(ax1, [min_l(3)-pad_l(3), max_l(3)+pad_l(3)]);
pbaspect(ax1, [1 1 1]);
axis(ax1, 'vis3d');
box(ax1, 'on');
set(ax1, 'FontSize', 7.5);
xlabel(ax1, sprintf('PC1 (%.1f%%)', explained(1)), 'FontSize', 8);
ylabel(ax1, sprintf('PC2 (%.1f%%)', explained(2)), 'FontSize', 8);
zlabel(ax1, sprintf('PC3 (%.1f%%)', explained(3)), 'FontSize', 8);
title(ax1, sprintf('1. Spazio Latente PCA Cubico: %d Features', length(lunar_feats)), 'FontSize', 9);
grid(ax1, 'on');
view(ax1, 38, 24);

lgd1 = legend(ax1, 'show');
set(lgd1, 'Position', [0.32, 0.65, 0.18, 0.13], 'FontSize', 6.8);

s_moon.DataTipTemplate.DataTipRows = [ ...
    dataTipTextRow('Target Lunare', lunar_names), ...
    dataTipTextRow('Miglior Analogo Terra', moon_best_names), ...
    dataTipTextRow('Distanza PCA', moon_to_earth_dist, '%.3f')];
s_best.DataTipTemplate.DataTipRows = [ ...
    dataTipTextRow('Miglior Match Terra', earth_names(best_earth_unique)), ...
    dataTipTextRow('Nazione', earth_country(best_earth_unique)), ...
    dataTipTextRow('Categoria', earth_luna_category(best_earth_unique)), ...
    dataTipTextRow('Equivalente Lunare', rev_lunar_names(best_earth_unique))];

% 3. Mappa 2D della Luna (In basso a sinistra)
ax2 = subplot('Position', [0.05, 0.06, 0.43, 0.35]);
im_m = image([-180 180], [90 -90], moon_img);
set(im_m, 'PickableParts', 'none');
set(gca, 'YDir', 'normal');
hold on;
p_moon = scatter(coords_Moon(:, 2), coords_Moon(:, 1), 180, 'y', 'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.8, 'DisplayName', 'Target Lunari');
for m = 1:N_moon
    text(coords_Moon(m, 2) + 5, coords_Moon(m, 1) + 5, lunar_names{m}, ...
        'FontSize', 7.5, 'FontWeight', 'bold', 'Color', [1 1 0.2], 'BackgroundColor', [0 0 0 0.65], 'Margin', 1, 'PickableParts', 'none');
end
xlim([-180 180]); ylim([-90 90]);
xticks(-180:60:180); yticks(-90:30:90);
grid on;
set(gca, 'GridColor', [0.8 0.8 0.8], 'GridAlpha', 0.25);
xlabel('Longitudine Lunare (°)'); ylabel('Latitudine Lunare (°)');
title('3. Mappa Lunare Panoramica');
legend(p_moon, 'Location', 'southwest');
p_moon.DataTipTemplate.DataTipRows = [dataTipTextRow('Target Lunare', lunar_names), dataTipTextRow('Miglior Analogo Terra', moon_best_names), dataTipTextRow('Lat Lunare', coords_Moon(:,1)), dataTipTextRow('Lon Lunare', coords_Moon(:,2))];

% 4. Mappa 2D della Terra (In basso a destra)
ax3 = subplot('Position', [0.52, 0.06, 0.44, 0.35]);
m_all = geoscatter(coords_Earth(:,1), coords_Earth(:,2), 45, [0.35 0.35 0.35], 'filled', 'DisplayName', 'Siti Terrestri');
ax3 = gca; % Salva il riferimento valido alla GeographicAxes
hold(ax3, 'on');
m_best = geoscatter(ax3, coords_Earth(best_earth_unique, 1), coords_Earth(best_earth_unique, 2), 160, 'r', 'filled', 'LineWidth', 2.0, 'DisplayName', 'Migliori Match');
try
    geobasemap(ax3, 'colorterrain');
    geolimits(ax3, [-85 82], [-170 175]);
catch
    geobasemap(ax3, 'landcover');
end
for i = 1:length(best_earth_unique)
    idx_s = best_earth_unique(i);
    text(ax3, coords_Earth(idx_s, 1) + 2.2, coords_Earth(idx_s, 2) + 2.0, earth_names{idx_s}, ...
        'FontSize', 8, 'FontWeight', 'bold', 'Color', [0.75 0.1 0.1], 'PickableParts', 'none');
end
title(ax3, '4. Mappa Topografica Terrestre');
legend(ax3, 'show', 'Location', 'southwest');
m_all.DataTipTemplate.DataTipRows = [dataTipTextRow('Sito Terrestre', earth_names), dataTipTextRow('Nazione', earth_country), dataTipTextRow('Categoria', earth_luna_category), dataTipTextRow('Equivalente Lunare', rev_lunar_names), dataTipTextRow('Latitudine', coords_Earth(:,1)), dataTipTextRow('Longitudine', coords_Earth(:,2))];
m_best.DataTipTemplate.DataTipRows = [dataTipTextRow('Miglior Match Terra', earth_names(best_earth_unique)), dataTipTextRow('Nazione', earth_country(best_earth_unique)), dataTipTextRow('Categoria', earth_luna_category(best_earth_unique)), dataTipTextRow('Equivalente Lunare', rev_lunar_names(best_earth_unique)), dataTipTextRow('Latitudine', coords_Earth(best_earth_unique,1)), dataTipTextRow('Longitudine', coords_Earth(best_earth_unique,2))];

% Collegamento Click per Radar Chart
set([s_moon, p_moon], 'ButtonDownFcn', @(src, evt) clickPlanetPoint(src, evt, ...
    moon_to_earth_best, moon_to_earth_dist, X_Moon_radar, X_Earth_radar, X_Moon, X_Earth, ...
    lunar_names, earth_names, clean_lunar_labels, ax_radar1, [0.95 0.75 0]));
set(m_all, 'ButtonDownFcn', @(src, evt) clickEarthPoint(src, evt, ...
    1:N_earth, earth_to_moon_best, earth_to_moon_dist, X_Moon_radar, X_Earth_radar, X_Moon, X_Earth, ...
    lunar_names, earth_names, clean_lunar_labels, ax_radar1, [0.95 0.75 0]));
set([s_best, m_best], 'ButtonDownFcn', @(src, evt) clickEarthPoint(src, evt, ...
    best_earth_unique, earth_to_moon_best, earth_to_moon_dist, X_Moon_radar, X_Earth_radar, X_Moon, X_Earth, ...
    lunar_names, earth_names, clean_lunar_labels, ax_radar1, [0.95 0.75 0]));

drawnow;
pause(0.5);

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

mars_names          = T_mars.Site_Name;
coords_Mars         = [T_mars.Latitude, T_mars.Longitude];
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

fprintf('-> [MARTE] Caricati %d target, %d analoghi terrestri e %d features quantitative.\n', ...
    N_mars, N_mars_earth, length(mars_feats));

% Standardizzazione & PCA (Marte)
all_data_mars = [X_Mars_Earth; X_Mars];
mu_m = mean(all_data_mars, 1);
sigma_m = std(all_data_mars, 0, 1);
sigma_m(sigma_m == 0) = 1e-5;
all_data_mars_norm = (all_data_mars - mu_m) ./ sigma_m;

X_Mars_Earth_norm = all_data_mars_norm(1:N_mars_earth, :);
X_Mars_norm       = all_data_mars_norm(N_mars_earth+1:end, :);

[~, S_m, V_m] = svd(all_data_mars_norm - mean(all_data_mars_norm, 1), 'econ');
score_all_m = all_data_mars_norm * V_m;

score_Mars_Earth_full = score_all_m(1:N_mars_earth, :);
score_Mars_full       = score_all_m(N_mars_earth+1:end, :);
score_Mars_Earth_3D   = score_all_m(1:N_mars_earth, 1:3);
score_Mars_3D         = score_all_m(N_mars_earth+1:end, 1:3);

eigVals_m = diag(S_m).^2 / (size(all_data_mars_norm, 1) - 1);
explained_mars = (eigVals_m / sum(eigVals_m)) * 100;

% K-Means Clustering Terrestre (Marte)
k_mars = 5;
rng(42);
centroids_m = X_Mars_Earth_norm(randperm(N_mars_earth, k_mars), :);
cluster_idx_mars = zeros(N_mars_earth, 1);
for iter = 1:100
    dists_m = zeros(N_mars_earth, k_mars);
    for c = 1:k_mars
        dists_m(:, c) = sum((X_Mars_Earth_norm - centroids_m(c, :)).^2, 2);
    end
    [~, cluster_idx_mars] = min(dists_m, [], 2);
    for c = 1:k_mars
        pts_m = X_Mars_Earth_norm(cluster_idx_mars == c, :);
        if ~isempty(pts_m)
            centroids_m(c, :) = mean(pts_m, 1);
        end
    end
end

% Matching Bidirezionale (Marte)
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

% Report Marte nella Command Window (Flusso C + Flusso D Inverso)
fprintf('\n================== FLUSSO C: TARGET MARZIANI -> ANALOGHI TERRESTRI ==================\n');
for m = 1:N_mars
    e_idx = mars_to_earth_best(m);
    fprintf('Target Marte: %s\n', mars_names{m});
    fprintf('   -> Miglior Analogo: %s (%s) | Distanza: %.3f | Cosine Sim: %.4f\n\n', ...
        mars_earth_names{e_idx}, mars_earth_country{e_idx}, mars_to_earth_dist(m), mars_to_earth_cos(m));
end

fprintf('\n================== FLUSSO D: ANALOGHI TERRESTRI -> TARGET MARZIANI ==================\n');
for e = 1:N_mars_earth
    m_idx = earth_to_mars_best(e);
    cos_sim_em = dot(score_Mars_Earth_full(e, :), score_Mars_full(m_idx, :)) / ...
                 (norm(score_Mars_Earth_full(e, :)) * norm(score_Mars_full(m_idx, :)));
    fprintf('Sito Terrestre: %s (%s) [%s]\n', mars_earth_names{e}, mars_earth_country{e}, mars_earth_category{e});
    fprintf('   -> Equivalente Marziano: %s | Distanza PCA: %.3f | Cosine Sim: %.4f\n\n', ...
        mars_names{m_idx}, earth_to_mars_dist(e), cos_sim_em);
end

% Mappa Marte
local_mars_file = 'mars_map.jpg';
if isfile(local_mars_file)
    mars_img = imread(local_mars_file);
else
    try
        mars_img = webread('https://upload.wikimedia.org/wikipedia/commons/thumb/0/02/OSIRIS_Mars_true_color.jpg/1280px-OSIRIS_Mars_true_color.jpg', weboptions('Timeout', 15, 'CertificateFilename', ''));
        imwrite(mars_img, local_mars_file);
    catch
        [Xg_m, Yg_m] = meshgrid(linspace(-180, 180, 1000), linspace(-90, 90, 500));
        m_red = 175 + 25 * sin(Xg_m * pi/80) .* cos(Yg_m * pi/40) + randn(500, 1000)*4;
        mars_img = cat(3, uint8(m_red), uint8(double(m_red)*0.52), uint8(double(m_red)*0.32));
    end
end

%% CRUSCOTTO GRAFICO FIGURA 2 (MARTE + CUBO 3D CENTRATO + RADAR CHART)
fig2 = figure('Name', 'NASA Planetary Analogue Discovery System - MARTE', 'Color', 'w', 'WindowState', 'maximized');
sgtitle('NASA Planetary Analogue Dual-Planet Discovery System (MARTE <-> TERRA)', 'FontSize', 13, 'FontWeight', 'bold');

% Normalizzazione [0, 1] per il Radar Chart di Marte
min_em = min(X_Mars_Earth, [], 1);
max_em = max(X_Mars_Earth, [], 1);
span_em = max_em - min_em;
span_em(span_em == 0) = 1;
X_Earth_Mars_radar = (X_Mars_Earth - min_em) ./ span_em;

min_mm = min(X_Mars, [], 1);
max_mm = max(X_Mars, [], 1);
span_mm = max_mm - min_mm;
span_mm(span_mm == 0) = 1;
X_Mars_radar = (X_Mars - min_mm) ./ span_mm;
clean_mars_labels = strrep(mars_feats, '_', ' ');

% 2. Radar Chart (In alto a destra)
ax_radar2 = polaraxes('Parent', fig2, 'Position', [0.58, 0.51, 0.34, 0.32]);
e1_mars = mars_to_earth_best(1);
updateRadarPlot(ax_radar2, X_Mars_radar(1,:), X_Earth_Mars_radar(e1_mars,:), ...
    X_Mars(1,:), X_Mars_Earth(e1_mars,:), ...
    mars_names{1}, mars_earth_names{e1_mars}, mars_to_earth_dist(1), clean_mars_labels, [1 0.45 0], 'r');

% 1. Spazio Latente 3D Cubico (In alto a sinistra)
ax1_m = subplot('Position', [0.10, 0.54, 0.19, 0.24]);
hold(ax1_m, 'on');
best_mars_earth_unique = unique(mars_to_earth_best);
mars_best_names = mars_earth_names(mars_to_earth_best);
rev_mars_names  = mars_names(earth_to_mars_best);

colori_cluster = parula(k_mars);
s_earth_m = gobjects(k_mars, 1);

for c = 1:k_mars
    idx_c = find(cluster_idx_mars == c);
    if isempty(idx_c), continue; end
    
    cats_c = strtrim(regexprep(mars_earth_category(idx_c), '\s*\(.*?\)', ''));
    [u_cats, ~, ic_cats] = unique(cats_c);
    counts = accumarray(ic_cats, 1);
    [~, sort_order] = sort(counts, 'descend');
    u_cats_sorted = u_cats(sort_order);
    
    if isscalar(u_cats_sorted)
        lbl_c = sprintf('C%d: %s', c, u_cats_sorted{1});
    else
        lbl_c = sprintf('C%d: %s / %s', c, u_cats_sorted{1}, u_cats_sorted{2});
    end
    
    s_earth_m(c) = scatter3(ax1_m, ...
        score_Mars_Earth_3D(idx_c, 1), score_Mars_Earth_3D(idx_c, 2), score_Mars_Earth_3D(idx_c, 3), ...
        45, colori_cluster(c, :), 'filled', 'MarkerEdgeColor', [0.2 0.2 0.2], 'LineWidth', 0.4, 'DisplayName', lbl_c);
    
    cluster_tag = repmat({lbl_c}, length(idx_c), 1);
    s_earth_m(c).DataTipTemplate.DataTipRows = [ ...
        dataTipTextRow('Sito Terrestre', mars_earth_names(idx_c)), ...
        dataTipTextRow('Nazione', mars_earth_country(idx_c)), ...
        dataTipTextRow('Cluster K-Means', cluster_tag), ...
        dataTipTextRow('Categoria CSV', mars_earth_category(idx_c)), ...
        dataTipTextRow('Equivalente Marziano', rev_mars_names(idx_c))];
        
    set(s_earth_m(c), 'ButtonDownFcn', @(src, evt) clickEarthPoint(src, evt, ...
        idx_c, earth_to_mars_best, earth_to_mars_dist, X_Mars_radar, X_Earth_Mars_radar, X_Mars, X_Mars_Earth, ...
        mars_names, mars_earth_names, clean_mars_labels, ax_radar2, [1 0.45 0]));
end

s_mars_pts = scatter3(ax1_m, score_Mars_3D(:,1), score_Mars_3D(:,2), score_Mars_3D(:,3), ...
    95, [1 0.35 0.1], 'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.2, 'DisplayName', 'Target Marziani');
s_best_m = scatter3(ax1_m, score_Mars_Earth_3D(best_mars_earth_unique, 1), score_Mars_Earth_3D(best_mars_earth_unique, 2), score_Mars_Earth_3D(best_mars_earth_unique, 3), ...
    85, 'r', 'LineWidth', 1.6, 'DisplayName', 'Migliori Match');

all_pts_mars = [score_Mars_Earth_3D; score_Mars_3D];
min_m = min(all_pts_mars, [], 1);
max_m = max(all_pts_mars, [], 1);
pad_m = (max_m - min_m) * 0.10;
xlim(ax1_m, [min_m(1)-pad_m(1), max_m(1)+pad_m(1)]);
ylim(ax1_m, [min_m(2)-pad_m(2), max_m(2)+pad_m(2)]);
zlim(ax1_m, [min_m(3)-pad_m(3), max_m(3)+pad_m(3)]);
pbaspect(ax1_m, [1 1 1]);
axis(ax1_m, 'vis3d');
box(ax1_m, 'on');
set(ax1_m, 'FontSize', 7.5);
xlabel(ax1_m, sprintf('PC1 (%.1f%%)', explained_mars(1)), 'FontSize', 8);
ylabel(ax1_m, sprintf('PC2 (%.1f%%)', explained_mars(2)), 'FontSize', 8);
zlabel(ax1_m, sprintf('PC3 (%.1f%%)', explained_mars(3)), 'FontSize', 8);
title(ax1_m, sprintf('1. Spazio Latente PCA Cubico: %d Features', length(mars_feats)), 'FontSize', 9);
grid(ax1_m, 'on');
view(ax1_m, 38, 24);

lgd2 = legend(ax1_m, 'show');
set(lgd2, 'Position', [0.32, 0.64, 0.18, 0.14], 'FontSize', 6.8);

s_mars_pts.DataTipTemplate.DataTipRows = [ ...
    dataTipTextRow('Target Marziano', mars_names), ...
    dataTipTextRow('Miglior Analogo Terra', mars_best_names), ...
    dataTipTextRow('Distanza PCA', mars_to_earth_dist, '%.3f')];
s_best_m.DataTipTemplate.DataTipRows = [ ...
    dataTipTextRow('Miglior Match Terra', mars_earth_names(best_mars_earth_unique)), ...
    dataTipTextRow('Nazione', mars_earth_country(best_mars_earth_unique)), ...
    dataTipTextRow('Equivalente Marziano', rev_mars_names(best_mars_earth_unique))];

% 3. Mappa 2D di Marte (In basso a sinistra)
ax2_m = subplot('Position', [0.05, 0.06, 0.43, 0.35]);
im_mars = image([-180 180], [90 -90], mars_img);
set(im_mars, 'PickableParts', 'none');
set(gca, 'YDir', 'normal');
hold on;
p_mars = scatter(coords_Mars(:, 2), coords_Mars(:, 1), 180, [1 0.85 0.1], 'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.8, 'DisplayName', 'Target Marziani');
for m = 1:N_mars
    text(coords_Mars(m, 2) + 5, coords_Mars(m, 1) + 5, mars_names{m}, ...
        'FontSize', 7, 'FontWeight', 'bold', 'Color', [1 0.95 0.3], 'BackgroundColor', [0 0 0 0.65], 'Margin', 1, 'PickableParts', 'none');
end
xlim([-180 180]); ylim([-90 90]);
xticks(-180:60:180); yticks(-90:30:90);
grid on;
set(gca, 'GridColor', [0.8 0.8 0.8], 'GridAlpha', 0.25);
xlabel('Longitudine Marziana (°)'); ylabel('Latitudine Marziana (°)');
title('3. Mappa Marziana Panoramica');
legend(p_mars, 'Location', 'southwest');
p_mars.DataTipTemplate.DataTipRows = [dataTipTextRow('Target Marziano', mars_names), dataTipTextRow('Miglior Analogo Terra', mars_best_names), dataTipTextRow('Lat Marziana', coords_Mars(:,1)), dataTipTextRow('Lon Marziana', coords_Mars(:,2))];

% 4. Mappa 2D della Terra (In basso a destra)
ax3_m = subplot('Position', [0.52, 0.06, 0.44, 0.35]);
m_all_m = geoscatter(coords_Mars_Earth(:,1), coords_Mars_Earth(:,2), 45, [0.25 0.25 0.25], 'filled', 'DisplayName', 'Siti Terrestri');
ax3_m = gca; % Salva il riferimento valido alla GeographicAxes
hold(ax3_m, 'on');
m_best_m = geoscatter(ax3_m, coords_Mars_Earth(best_mars_earth_unique, 1), coords_Mars_Earth(best_mars_earth_unique, 2), 160, 'r', 'filled', 'LineWidth', 2.0, 'DisplayName', 'Migliori Match');
try
    geobasemap(ax3_m, 'colorterrain');
    geolimits(ax3_m, [-55 82], [-145 155]);
catch
    geobasemap(ax3_m, 'landcover');
end
for i = 1:length(best_mars_earth_unique)
    idx_s = best_mars_earth_unique(i);
    text(ax3_m, coords_Mars_Earth(idx_s, 1) + 2.5, coords_Mars_Earth(idx_s, 2) + 2.0, mars_earth_names{idx_s}, ...
        'FontSize', 8, 'FontWeight', 'bold', 'Color', [0.75 0.05 0.05], 'PickableParts', 'none');
end
title(ax3_m, '4. Mappa Topografica Terrestre');
legend(ax3_m, 'show', 'Location', 'southwest');
m_all_m.DataTipTemplate.DataTipRows = [dataTipTextRow('Sito Terrestre', mars_earth_names), dataTipTextRow('Nazione', mars_earth_country), dataTipTextRow('Categoria', mars_earth_category), dataTipTextRow('Equivalente Marziano', rev_mars_names), dataTipTextRow('Latitudine', coords_Mars_Earth(:,1)), dataTipTextRow('Longitudine', coords_Mars_Earth(:,2))];
m_best_m.DataTipTemplate.DataTipRows = [dataTipTextRow('Miglior Match Terra', mars_earth_names(best_mars_earth_unique)), dataTipTextRow('Nazione', mars_earth_country(best_mars_earth_unique)), dataTipTextRow('Categoria', mars_earth_category(best_mars_earth_unique)), dataTipTextRow('Equivalente Marziano', rev_mars_names(best_mars_earth_unique)), dataTipTextRow('Latitudine', coords_Mars_Earth(best_mars_earth_unique,1)), dataTipTextRow('Longitudine', coords_Mars_Earth(best_mars_earth_unique,2))];

% Collegamento Click per Radar Chart
set([s_mars_pts, p_mars], 'ButtonDownFcn', @(src, evt) clickPlanetPoint(src, evt, ...
    mars_to_earth_best, mars_to_earth_dist, X_Mars_radar, X_Earth_Mars_radar, X_Mars, X_Mars_Earth, ...
    mars_names, mars_earth_names, clean_mars_labels, ax_radar2, [1 0.45 0]));
set(m_all_m, 'ButtonDownFcn', @(src, evt) clickEarthPoint(src, evt, ...
    1:N_mars_earth, earth_to_mars_best, earth_to_mars_dist, X_Mars_radar, X_Earth_Mars_radar, X_Mars, X_Mars_Earth, ...
    mars_names, mars_earth_names, clean_mars_labels, ax_radar2, [1 0.45 0]));
set([s_best_m, m_best_m], 'ButtonDownFcn', @(src, evt) clickEarthPoint(src, evt, ...
    best_mars_earth_unique, earth_to_mars_best, earth_to_mars_dist, X_Mars_radar, X_Earth_Mars_radar, X_Mars, X_Mars_Earth, ...
    mars_names, mars_earth_names, clean_mars_labels, ax_radar2, [1 0.45 0]));

drawnow;

%% ========================================================================================
%% PARTE III: RICERCA SITI TERRESTRI SINGOLO O MULTI-TEST (MULTI-FEATURE ENGINEERING)
%% ========================================================================================
fprintf('\n========================================================================================\n');
fprintf('     MODULO SELEZIONE SITI TERRESTRI PER TEST SINGOLI O COMBINATI (MULTI-TEST)          \n');
fprintf('========================================================================================\n');

h_test_luna = gobjects(0);
h_test_mars = gobjects(0);

while true
    pianeta_scelto = input('\nSeleziona il corpo celeste (1 = LUNA, 2 = MARTE, 0 = ESCI): ');
    
    if isempty(pianeta_scelto)
        continue;
    elseif pianeta_scelto == 0
        if ~isempty(h_test_luna) && any(isgraphics(h_test_luna))
            delete(h_test_luna(isgraphics(h_test_luna)));
            if exist('ax3', 'var') && isgraphics(ax3), legend(ax3, 'show', 'Location', 'southwest'); end
        end
        if ~isempty(h_test_mars) && any(isgraphics(h_test_mars))
            delete(h_test_mars(isgraphics(h_test_mars)));
            if exist('ax3_m', 'var') && isgraphics(ax3_m), legend(ax3_m, 'show', 'Location', 'southwest'); end
        end
        drawnow;
        fprintf('Stelline rimosse dalle mappe. Uscita dal modulo di ricerca test.\n');
        break;
        
    elseif pianeta_scelto == 1
        fprintf('\nTest disponibili per LUNA (puoi sceglierne uno o piu insieme, es: 1 2 4 oppure [1, 4]):\n');
        fprintf('  1 - Atterraggio / Touchdown (Area >= 200k m2, Slope <= 5 deg, Massi <= 0.35m, Portanza >= 5 kPa)\n');
        fprintf('  2 - Mobilita Rover (Slope <= 8 deg, Massi <= 0.38m, Coesione >= 0.44 kPa, Inerzia Termica >= 200)\n');
        fprintf('  3 - Intera Base Lunare (Slope25m <= 4.6 deg, Area >= 400k m2, Portanza >= 5.5 kPa, Sole >= 70%%, DTE >= 90%%, Sep >= 2000m)\n');
        fprintf('  4 - Singolo Modulo Abitativo 2-4 Persone (Stesse features Base: Slope25m < 7 deg, Area >= 230k m2, Portanza >= 5.0 kPa, Sep >= 1400m)\n');
        fprintf('  5 - EVA / Tubi di Lava (Diametro >= 35m, Profondita >= 20m, Coesione >= 0.44 kPa, dT >= 35 K)\n');
        fprintf('  6 - Sistemi Energetici e ISRU (Sole >= 70%%, FeO/TiO2 o Ghiaccio, Scavabilita <= 1710 kg/m3)\n');
        raw_in = input('Inserisci il numero (o i numeri) dei test lunari (1-6): ', 's');
        test_scelti = unique(str2num(raw_in), 'stable'); %#ok<ST2NM>
        
        if isempty(test_scelti) || any(test_scelti < 1 | test_scelti > 6)
            fprintf('Scelta non valida. Inserisci numeri compresi tra 1 e 6 (es. 1 2 4).\n');
            continue;
        end
        
        num_t = length(test_scelti);
        masks_mat  = false(N_earth, num_t);
        scores_mat = zeros(N_earth, num_t);
        nomi_scelti = cell(1, num_t);
        
        for k_t = 1:num_t
            t_id = test_scelti(k_t);
            switch t_id
                case 1 % Atterraggio / Touchdown
                    m_k = (T_earth.Safe_Area_m2 >= 200000) & (T_earth.Slope_5m <= 5.0) & (T_earth.Slope_1m <= 4.0) & ...
                          (T_earth.Hab_Sep_m >= 2000) & (T_earth.Boulder_Height_m <= 0.35) & ...
                          (T_earth.Boulder_Area_frac <= 0.05) & (T_earth.Bearing_Capacity_kPa >= 5.0);
                    s_k = (T_earth.Safe_Area_m2 ./ max(T_earth.Safe_Area_m2)) * 30 + ...
                          (1 - T_earth.Slope_5m / 10) * 25 + ...
                          (1 - T_earth.Boulder_Area_frac / 0.10) * 25 + ...
                          (T_earth.Bearing_Capacity_kPa ./ max(T_earth.Bearing_Capacity_kPa)) * 20;
                    n_k = 'Atterraggio';
                    
                case 2 % Mobilità Rover
                    m_k = (T_earth.Traverse_Slope_deg <= 8.0) & (T_earth.Friction_Angle_deg >= 38.0) & ...
                          (T_earth.Boulder_Height_m <= 0.38) & (T_earth.Cohesion_kPa >= 0.44) & ...
                          (T_earth.Thermal_Inertia >= 200);
                    s_k = (1 - T_earth.Traverse_Slope_deg / 15) * 35 + ...
                          (1 - T_earth.Boulder_Height_m / 0.50) * 25 + ...
                          (T_earth.Cohesion_kPa ./ max(T_earth.Cohesion_kPa)) * 20 + ...
                          (T_earth.Thermal_Inertia ./ max(T_earth.Thermal_Inertia)) * 20;
                    n_k = 'Rover';
                    
                case 3 % Intera Base Lunare
                    m_k = (T_earth.Slope_25m <= 4.6) & (T_earth.Safe_Area_m2 >= 400000) & ...
                          (T_earth.Bearing_Capacity_kPa >= 5.5) & (T_earth.Sun_Visibility_pct >= 70) & ...
                          (T_earth.DTE_pct >= 90) & (T_earth.Hab_Sep_m >= 2000);
                    s_k = (T_earth.Bearing_Capacity_kPa ./ max(T_earth.Bearing_Capacity_kPa)) * 30 + ...
                          (T_earth.Sun_Visibility_pct / 100) * 30 + ...
                          (T_earth.Hab_Sep_m ./ max(T_earth.Hab_Sep_m)) * 25 + ...
                          (1 - T_earth.Slope_25m / 10) * 15;
                    n_k = 'Intera Base';
                    
                case 4 % Singolo Modulo Abitativo (2-4 persone)
                    m_k = (T_earth.Slope_25m < 7.0) & (T_earth.Safe_Area_m2 >= 230000) & ...
                          (T_earth.Bearing_Capacity_kPa >= 5.0) & (T_earth.Sun_Visibility_pct >= 60) & ...
                          (T_earth.DTE_pct >= 90) & (T_earth.Hab_Sep_m >= 1400);
                    s_k = (T_earth.Bearing_Capacity_kPa ./ max(T_earth.Bearing_Capacity_kPa)) * 30 + ...
                          (T_earth.Sun_Visibility_pct / 100) * 30 + ...
                          (T_earth.Hab_Sep_m ./ max(T_earth.Hab_Sep_m)) * 25 + ...
                          (1 - T_earth.Slope_25m / 10) * 15;
                    n_k = 'Modulo 2-4p';
                    
                case 5 % EVA / Tubi di Lava
                    dT = T_earth.T_max_K - T_earth.T_min_K;
                    m_k = (T_earth.Pit_Depth_m >= 20) & (T_earth.Pit_Diameter_m >= 35) & ...
                          (T_earth.Cohesion_kPa >= 0.44) & (dT >= 35);
                    s_k = min(T_earth.Pit_Depth_m / 50, 1) * 40 + ...
                          min(dT / 60, 1) * 35 + ...
                          (T_earth.Cohesion_kPa ./ max(T_earth.Cohesion_kPa)) * 25;
                    n_k = 'EVA/Tubi Lava';
                    
                case 6 % Energia e ISRU
                    has_resource = ((T_earth.FeO_wt_pct >= 13.5) & (T_earth.TiO2_wt_pct >= 3.0)) | (T_earth.Water_Ice_wt_pct > 0);
                    m_k = (T_earth.Sun_Visibility_pct >= 70.0) & has_resource & ...
                          (T_earth.Density_0_60cm_kg_m3 <= 1710) & (T_earth.Slope_5m <= 6.5);
                    res_score = max(T_earth.FeO_wt_pct / 16, T_earth.Water_Ice_wt_pct / 5);
                    s_k = (T_earth.Sun_Visibility_pct / 100) * 45 + ...
                          min(res_score, 1) * 35 + ...
                          (1 - T_earth.Slope_5m / 10) * 20;
                    n_k = 'Energia/ISRU';
            end
            masks_mat(:, k_t)  = m_k;
            scores_mat(:, k_t) = s_k;
            nomi_scelti{k_t}   = n_k;
        end
        
        test_nome  = strjoin(nomi_scelti, ' + ');
        num_passed = sum(masks_mat, 2);
        score_comb = mean(scores_mat, 2);
        mask_all   = all(masks_mat, 2);
        
        if any(mask_all)
            idx_validi = find(mask_all);
            [~, ord] = sort(score_comb(idx_validi), 'descend');
            idx_validi = idx_validi(ord);
            fprintf('\n>>> RISULTATI MULTI-TEST "%s" (%d siti soddisfano TUTTI i %d test su %d):\n', ...
                test_nome, length(idx_validi), num_t, N_earth);
        else
            % Modalità Best-Fit se si combinano test fisicamente opposti
            max_p = max(num_passed);
            idx_validi = find(num_passed == max_p);
            [~, ord] = sort(score_comb(idx_validi), 'descend');
            idx_validi = idx_validi(ord(1:min(5, end)));
            fprintf('\n>>> Nessun sito soddisfa contemporaneamente tutti i %d test.\n', num_t);
            fprintf('>>> Ecco i migliori siti multi-ruolo che superano %d/%d test per "%s":\n', max_p, num_t, test_nome);
        end
        
        for i = 1:length(idx_validi)
            id = idx_validi(i);
            fprintf('  %d) %s (%s) | Test superati: %d/%d | Score Medio: %.1f/100 | SafeArea: %.0f m2 | Slope25m: %.1f deg | Portanza: %.1f kPa\n', ...
                i, earth_names{id}, earth_country{id}, num_passed(id), num_t, score_comb(id), ...
                T_earth.Safe_Area_m2(id), T_earth.Slope_25m(id), T_earth.Bearing_Capacity_kPa(id));
        end
        
        if ~isempty(h_test_luna) && any(isgraphics(h_test_luna))
            delete(h_test_luna(isgraphics(h_test_luna)));
        end
        
        if ~isempty(idx_validi)
            if ~exist('ax3', 'var') || ~isgraphics(ax3)
                figure('Name', ['Risultati Test: ' test_nome], 'Color', 'w');
                geoplot(coords_Earth(:,1), coords_Earth(:,2), 'o', 'Color', [0.4 0.4 0.4], 'DisplayName', 'Tutti i siti');
                ax3 = gca;
                geobasemap(ax3, 'colorterrain');
            end
            hold(ax3, 'on');
            h_test_luna = geoscatter(ax3, coords_Earth(idx_validi, 1), coords_Earth(idx_validi, 2), 220, 'g', 'p', 'filled', ...
                'MarkerEdgeColor', 'k', 'LineWidth', 1.2, 'DisplayName', ['Idonei: ' test_nome]);
            legend(ax3, 'show', 'Location', 'southwest');
            figure(ancestor(ax3, 'figure'));
        end

    elseif pianeta_scelto == 2
        fprintf('\nTest disponibili per MARTE (puoi sceglierne uno o piu insieme, es: 1 2 4 oppure [1, 4]):\n');
        fprintf('  1 - Atterraggio / EDL (Area >= 500k m2, Slope <= 10 deg, Inerzia Termica >= 150, Albedo <= 0.26)\n');
        fprintf('  2 - Mobilita Rover (Massi <= 7%%, Inerzia Termica >= 150, Terreni basaltici/desertici compatti)\n');
        fprintf('  3 - Intera Base Marziana (Suolo solfati/basalti compatti, Area >= 500k m2, Slope <= 5 deg, Inerzia >= 180)\n');
        fprintf('  4 - Singolo Modulo Abitativo 2-4 Persone (Stesse features Base: Area >= 200k m2, Slope <= 8 deg, Inerzia >= 160)\n');
        fprintf('  5 - Test Droni / UAV (Alta quota >= 2500 m per aria rarefatta, Vento <= 10 m/s, Terreno aperto)\n');
        fprintf('  6 - Test EVA Estremi (Temperatura min <= -18 C / 255 K, Aria iper-secca <= 15%% HR, Polvere fine)\n');
        fprintf('  7 - Sistemi Energetici e ISRU (Ghiaccio/Permafrost o Minerali Idrati, Scavabilita <= 2 m)\n');
        raw_in = input('Inserisci il numero (o i numeri) dei test marziani (1-7): ', 's');
        test_scelti = unique(str2num(raw_in), 'stable'); %#ok<ST2NM>
        
        if isempty(test_scelti) || any(test_scelti < 1 | test_scelti > 7)
            fprintf('Scelta non valida. Inserisci numeri compresi tra 1 e 7 (es. 1 2 4).\n');
            continue;
        end
        
        cat_m = lower(string(mars_earth_category));
        vars_m = T_mars_earth.Properties.VariableNames;
        num_t = length(test_scelti);
        masks_mat  = false(N_mars_earth, num_t);
        scores_mat = zeros(N_mars_earth, num_t);
        nomi_scelti = cell(1, num_t);
        
        for k_t = 1:num_t
            t_id = test_scelti(k_t);
            m_k = true(N_mars_earth, 1);
            s_k = ones(N_mars_earth, 1) * 60;
            
            switch t_id
                case 1 % Atterraggio / EDL
                    m_k = contains(cat_m, 'desert') | contains(cat_m, 'salt') | contains(cat_m, 'plain') | contains(cat_m, 'plateau');
                    if ismember('Safe_Area_m2', vars_m), m_k = m_k & (T_mars_earth.Safe_Area_m2 >= 500000); end
                    if ismember('Thermal_Inertia', vars_m)
                        m_k = m_k & (T_mars_earth.Thermal_Inertia >= 150);
                        s_k = s_k + (T_mars_earth.Thermal_Inertia ./ max(T_mars_earth.Thermal_Inertia)) * 25;
                    end
                    if ismember('Albedo', vars_m), m_k = m_k & (T_mars_earth.Albedo <= 0.26); end
                    if ismember('Slope_deg', vars_m)
                        m_k = m_k & (T_mars_earth.Slope_deg <= 10);
                        s_k = s_k + (1 - T_mars_earth.Slope_deg / 15) * 15;
                    end
                    n_k = 'EDL';
                    
                case 2 % Mobilità Rover
                    m_k = (contains(cat_m, 'volcanic') | contains(cat_m, 'desert') | contains(cat_m, 'dune') | contains(cat_m, 'plateau')) & ...
                          ~contains(cat_m, 'glacio') & ~contains(cat_m, 'polar coastal');
                    if ismember('Boulder_Area_frac', vars_m), m_k = m_k & (T_mars_earth.Boulder_Area_frac <= 0.07); end
                    if ismember('Thermal_Inertia', vars_m)
                        m_k = m_k & (T_mars_earth.Thermal_Inertia >= 150);
                        s_k = s_k + (T_mars_earth.Thermal_Inertia ./ max(T_mars_earth.Thermal_Inertia)) * 30;
                    end
                    if ismember('Traverse_Slope_deg', vars_m), m_k = m_k & (T_mars_earth.Traverse_Slope_deg <= 20); end
                    n_k = 'Rover';
                    
                case 3 % Intera Base Marziana
                    m_k = (contains(cat_m, 'salt') | contains(cat_m, 'volcanic') | contains(cat_m, 'desert') | contains(cat_m, 'plateau')) & ...
                          ~contains(cat_m, 'mobile dunes') & ~contains(cat_m, 'glacio') & ~contains(cat_m, 'fumaroles') & ~contains(cat_m, 'high-altitude');
                    if ismember('Safe_Area_m2', vars_m), m_k = m_k & (T_mars_earth.Safe_Area_m2 >= 500000); end
                    if ismember('Slope_deg', vars_m)
                        m_k = m_k & (T_mars_earth.Slope_deg <= 5);
                        s_k = s_k + (1 - T_mars_earth.Slope_deg / 12) * 20;
                    elseif ismember('Mean_Slope_deg', vars_m)
                        m_k = m_k & (T_mars_earth.Mean_Slope_deg <= 5);
                        s_k = s_k + (1 - T_mars_earth.Mean_Slope_deg / 12) * 20;
                    end
                    if ismember('Thermal_Inertia', vars_m)
                        m_k = m_k & (T_mars_earth.Thermal_Inertia >= 180);
                        s_k = s_k + (T_mars_earth.Thermal_Inertia ./ max(T_mars_earth.Thermal_Inertia)) * 20;
                    end
                    if ismember('Bearing_Capacity_kPa', vars_m), m_k = m_k & (T_mars_earth.Bearing_Capacity_kPa >= 5.5); end
                    n_k = 'Intera Base';
                    
                case 4 % Singolo Modulo Abitativo (2-4 persone)
                    m_k = (contains(cat_m, 'salt') | contains(cat_m, 'volcanic') | contains(cat_m, 'desert') | contains(cat_m, 'plateau')) & ...
                          ~contains(cat_m, 'mobile dunes') & ~contains(cat_m, 'glacio') & ~contains(cat_m, 'fumaroles');
                    if ismember('Safe_Area_m2', vars_m), m_k = m_k & (T_mars_earth.Safe_Area_m2 >= 200000); end
                    if ismember('Slope_deg', vars_m)
                        m_k = m_k & (T_mars_earth.Slope_deg <= 8);
                        s_k = s_k + (1 - T_mars_earth.Slope_deg / 12) * 20;
                    elseif ismember('Mean_Slope_deg', vars_m)
                        m_k = m_k & (T_mars_earth.Mean_Slope_deg <= 8);
                        s_k = s_k + (1 - T_mars_earth.Mean_Slope_deg / 12) * 20;
                    end
                    if ismember('Thermal_Inertia', vars_m)
                        m_k = m_k & (T_mars_earth.Thermal_Inertia >= 160);
                        s_k = s_k + (T_mars_earth.Thermal_Inertia ./ max(T_mars_earth.Thermal_Inertia)) * 20;
                    end
                    if ismember('Bearing_Capacity_kPa', vars_m), m_k = m_k & (T_mars_earth.Bearing_Capacity_kPa >= 4.5); end
                    n_k = 'Modulo 2-4p';
                    
                case 5 % Test Droni / UAV
                    m_k = contains(cat_m, 'mountain') | contains(cat_m, 'high-altitude') | contains(cat_m, 'plateau');
                    if ismember('Elevation_m', vars_m)
                        m_k = m_k & (T_mars_earth.Elevation_m >= 2500);
                        s_k = s_k + (T_mars_earth.Elevation_m ./ max(T_mars_earth.Elevation_m)) * 35;
                    end
                    if ismember('Wind_Speed_ms', vars_m), m_k = m_k & (T_mars_earth.Wind_Speed_ms <= 10); end
                    n_k = 'Droni/UAV';
                    
                case 6 % Test EVA / Freddo e Aridità Estrema
                    m_k = contains(cat_m, 'cold') | contains(cat_m, 'polar') | contains(cat_m, 'arctic') | contains(cat_m, 'hyper-arid');
                    if ismember('T_min_K', vars_m)
                        m_k = m_k & (T_mars_earth.T_min_K <= 255);
                        s_k = s_k + (1 - (T_mars_earth.T_min_K - 200) / 100) * 35;
                    end
                    if ismember('Humidity_pct', vars_m), m_k = m_k & (T_mars_earth.Humidity_pct <= 15); end
                    n_k = 'EVA Estremo';
                    
                case 7 % Energia e ISRU
                    m_k = (contains(cat_m, 'polar') | contains(cat_m, 'periglacial') | contains(cat_m, 'salt') | contains(cat_m, 'gypsum')) & ...
                          ~contains(cat_m, 'basaltic flows');
                    if ismember('Water_Ice_wt_pct', vars_m)
                        m_k = m_k | (T_mars_earth.Water_Ice_wt_pct > 0);
                        s_k = s_k + min(T_mars_earth.Water_Ice_wt_pct / 5, 1) * 35;
                    end
                    n_k = 'ISRU/Ghiaccio';
            end
            masks_mat(:, k_t)  = m_k;
            scores_mat(:, k_t) = min(s_k, 100);
            nomi_scelti{k_t}   = n_k;
        end
        
        test_nome  = strjoin(nomi_scelti, ' + ');
        num_passed = sum(masks_mat, 2);
        score_comb = mean(scores_mat, 2);
        mask_all   = all(masks_mat, 2);
        
        if any(mask_all)
            idx_validi = find(mask_all);
            [~, ord] = sort(score_comb(idx_validi), 'descend');
            idx_validi = idx_validi(ord);
            fprintf('\n>>> RISULTATI MULTI-TEST "%s" (%d siti soddisfano TUTTI i %d test su %d):\n', ...
                test_nome, length(idx_validi), num_t, N_mars_earth);
        else
            max_p = max(num_passed);
            idx_validi = find(num_passed == max_p);
            [~, ord] = sort(score_comb(idx_validi), 'descend');
            idx_validi = idx_validi(ord(1:min(5, end)));
            fprintf('\n>>> Nessun sito soddisfa contemporaneamente tutti i %d test.\n', num_t);
            fprintf('>>> Ecco i migliori siti multi-ruolo che superano %d/%d test per "%s":\n', max_p, num_t, test_nome);
        end
        
        for i = 1:length(idx_validi)
            id = idx_validi(i);
            fprintf('  %d) %s (%s) | Test superati: %d/%d | Score Medio: %.1f/100 | Categoria: %s | Lat: %.2f, Lon: %.2f\n', ...
                i, mars_earth_names{id}, mars_earth_country{id}, num_passed(id), num_t, score_comb(id), ...
                mars_earth_category{id}, coords_Mars_Earth(id,1), coords_Mars_Earth(id,2));
        end
        
        if ~isempty(h_test_mars) && any(isgraphics(h_test_mars))
            delete(h_test_mars(isgraphics(h_test_mars)));
        end
        
        if ~isempty(idx_validi)
            if ~exist('ax3_m', 'var') || ~isgraphics(ax3_m)
                figure('Name', ['Risultati Test: ' test_nome], 'Color', 'w');
                geoplot(coords_Mars_Earth(:,1), coords_Mars_Earth(:,2), 'o', 'Color', [0.4 0.4 0.4], 'DisplayName', 'Tutti i siti');
                ax3_m = gca;
                geobasemap(ax3_m, 'colorterrain');
            end
            hold(ax3_m, 'on');
            h_test_mars = geoscatter(ax3_m, coords_Mars_Earth(idx_validi, 1), coords_Mars_Earth(idx_validi, 2), 220, 'c', 'p', 'filled', ...
                'MarkerEdgeColor', 'k', 'LineWidth', 1.2, 'DisplayName', ['Idonei: ' test_nome]);
            legend(ax3_m, 'show', 'Location', 'southwest');
            figure(ancestor(ax3_m, 'figure'));
        end
    else
        fprintf('Scelta non valida. Inserisci 1, 2 oppure 0.\n');
    end
end

%% ========================================================================================
%% FUNZIONI AUSILIARIE PER IL RADAR CHART CON TOOLTIP DELLE FEATURES
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
