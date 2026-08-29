#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <string.h>

static inline npy_intp index3(npy_intp x, npy_intp y, npy_intp z, npy_intp sy, npy_intp sz) {
    return (x * sy + y) * sz + z;
}

static inline npy_bool get3(npy_bool *data, npy_intp sx, npy_intp sy, npy_intp sz, npy_intp x, npy_intp y, npy_intp z) {
    if (x < 0 || y < 0 || z < 0 || x >= sx || y >= sy || z >= sz) {
        return 0;
    }
    return data[index3(x, y, z, sy, sz)];
}

static inline void c5_offset(int one_based_index, int *dx, int *dy, int *dz) {
    int idx = one_based_index - 1;
    *dx = (idx % 5) - 2;
    *dy = ((idx / 5) % 5) - 2;
    *dz = (idx / 25) - 2;
}

static inline npy_bool c5(npy_bool *data, npy_intp sx, npy_intp sy, npy_intp sz, npy_intp x, npy_intp y, npy_intp z, int one_based_index) {
    int dx, dy, dz;
    c5_offset(one_based_index, &dx, &dy, &dz);
    return get3(data, sx, sy, sz, x + dx, y + dy, z + dz);
}

static int border_point_type_c(npy_bool *data, npy_intp sx, npy_intp sy, npy_intp sz, npy_intp x, npy_intp y, npy_intp z) {
    int sopen_indices[] = {38, 58, 62, 64, 68, 88};
    for (int i = 0; i < 6; i++) {
        if (!c5(data, sx, sy, sz, x, y, z, sopen_indices[i])) {
            return 1;
        }
    }

    if (
        (!c5(data, sx, sy, sz, x, y, z, 33) && c5(data, sx, sy, sz, x, y, z, 13) && c5(data, sx, sy, sz, x, y, z, 53)) ||
        (!c5(data, sx, sy, sz, x, y, z, 37) && c5(data, sx, sy, sz, x, y, z, 13) && c5(data, sx, sy, sz, x, y, z, 61)) ||
        (!c5(data, sx, sy, sz, x, y, z, 39) && c5(data, sx, sy, sz, x, y, z, 13) && c5(data, sx, sy, sz, x, y, z, 65)) ||
        (!c5(data, sx, sy, sz, x, y, z, 43) && c5(data, sx, sy, sz, x, y, z, 13) && c5(data, sx, sy, sz, x, y, z, 73)) ||
        (!c5(data, sx, sy, sz, x, y, z, 57) && c5(data, sx, sy, sz, x, y, z, 53) && c5(data, sx, sy, sz, x, y, z, 61)) ||
        (!c5(data, sx, sy, sz, x, y, z, 59) && c5(data, sx, sy, sz, x, y, z, 53) && c5(data, sx, sy, sz, x, y, z, 65)) ||
        (!c5(data, sx, sy, sz, x, y, z, 67) && c5(data, sx, sy, sz, x, y, z, 61) && c5(data, sx, sy, sz, x, y, z, 73)) ||
        (!c5(data, sx, sy, sz, x, y, z, 69) && c5(data, sx, sy, sz, x, y, z, 73) && c5(data, sx, sy, sz, x, y, z, 65)) ||
        (!c5(data, sx, sy, sz, x, y, z, 83) && c5(data, sx, sy, sz, x, y, z, 113) && c5(data, sx, sy, sz, x, y, z, 53)) ||
        (!c5(data, sx, sy, sz, x, y, z, 87) && c5(data, sx, sy, sz, x, y, z, 113) && c5(data, sx, sy, sz, x, y, z, 61)) ||
        (!c5(data, sx, sy, sz, x, y, z, 89) && c5(data, sx, sy, sz, x, y, z, 113) && c5(data, sx, sy, sz, x, y, z, 65)) ||
        (!c5(data, sx, sy, sz, x, y, z, 93) && c5(data, sx, sy, sz, x, y, z, 113) && c5(data, sx, sy, sz, x, y, z, 73))
    ) {
        return 2;
    }

    if (
        (!c5(data, sx, sy, sz, x, y, z, 32) && c5(data, sx, sy, sz, x, y, z, 13) && c5(data, sx, sy, sz, x, y, z, 53) && c5(data, sx, sy, sz, x, y, z, 61)) ||
        (!c5(data, sx, sy, sz, x, y, z, 34) && c5(data, sx, sy, sz, x, y, z, 13) && c5(data, sx, sy, sz, x, y, z, 53) && c5(data, sx, sy, sz, x, y, z, 65)) ||
        (!c5(data, sx, sy, sz, x, y, z, 42) && c5(data, sx, sy, sz, x, y, z, 13) && c5(data, sx, sy, sz, x, y, z, 73) && c5(data, sx, sy, sz, x, y, z, 61)) ||
        (!c5(data, sx, sy, sz, x, y, z, 44) && c5(data, sx, sy, sz, x, y, z, 13) && c5(data, sx, sy, sz, x, y, z, 73) && c5(data, sx, sy, sz, x, y, z, 65)) ||
        (!c5(data, sx, sy, sz, x, y, z, 82) && c5(data, sx, sy, sz, x, y, z, 113) && c5(data, sx, sy, sz, x, y, z, 53) && c5(data, sx, sy, sz, x, y, z, 61)) ||
        (!c5(data, sx, sy, sz, x, y, z, 84) && c5(data, sx, sy, sz, x, y, z, 113) && c5(data, sx, sy, sz, x, y, z, 53) && c5(data, sx, sy, sz, x, y, z, 65)) ||
        (!c5(data, sx, sy, sz, x, y, z, 92) && c5(data, sx, sy, sz, x, y, z, 113) && c5(data, sx, sy, sz, x, y, z, 73) && c5(data, sx, sy, sz, x, y, z, 61)) ||
        (!c5(data, sx, sy, sz, x, y, z, 94) && c5(data, sx, sy, sz, x, y, z, 113) && c5(data, sx, sy, sz, x, y, z, 73) && c5(data, sx, sy, sz, x, y, z, 65))
    ) {
        return 3;
    }
    return 0;
}

static int endpoint_c(npy_bool *data, npy_intp sx, npy_intp sy, npy_intp sz, npy_intp x, npy_intp y, npy_intp z) {
    int count = 0;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                count += get3(data, sx, sy, sz, x + dx, y + dy, z + dz) ? 1 : 0;
            }
        }
    }
    return count <= 2;
}

static int encircles4(const npy_bool cfg[125]) {
    int sets[8][12] = {
        {1,2,3,4,5,8,9,12,13,14,15,16},
        {1,2,3,5,7,8,9,12,13,14,15,16},
        {1,2,3,5,7,8,9,10,12,14,15,16},
        {5,6,7,8,9,12,13,14,15,16,0,0},
        {1,2,3,5,7,8,9,12,13,14,15,16},
        {2,3,4,5,6,8,9,12,13,14,15,16},
        {6,7,8,10,12,14,15,16,0,0,0,0},
        {2,3,4,6,8,10,12,14,15,16,0,0}
    };
    for (int s = 0; s < 8; s++) {
        int any_value = 0;
        for (int j = 0; j < 12; j++) {
            int idx = sets[s][j];
            if (idx == 0) {
                continue;
            }
            int zero_based = idx - 1;
            int y = zero_based % 4;
            int z = zero_based / 4;
            int cfg_idx = (2 * 5 + y) * 5 + z;
            if (cfg[cfg_idx]) {
                any_value = 1;
                break;
            }
        }
        if (!any_value) {
            return 1;
        }
    }
    return 0;
}

static inline npy_bool cfg5_index(const npy_bool cfg[125], int one_based_index) {
    int idx = one_based_index - 1;
    int x = idx % 5;
    int y = (idx / 5) % 5;
    int z = idx / 25;
    return cfg[(x * 5 + y) * 5 + z];
}

static int c12_reference_c(const npy_bool cfg[125]) {
    int n_sum_yz = 0;
    int n0_sum = 0;
    int n2_sum = 0;
    for (int y = 1; y <= 3; y++) {
        for (int z = 1; z <= 3; z++) {
            int any_x = 0;
            for (int x = 1; x <= 3; x++) {
                npy_bool value = cfg[(x * 5 + y) * 5 + z];
                any_x |= value ? 1 : 0;
                if (x == 1 && value) n0_sum++;
                if (x == 3 && value) n2_sum++;
            }
            if (any_x) n_sum_yz++;
        }
    }
    int cond1 = encircles4(cfg) && n0_sum > 0 && n2_sum > 0;
    int cond2 = (!cfg5_index(cfg, 62)) && ((!cfg5_index(cfg, 64)) || (!cfg5_index(cfg, 65))) && n_sum_yz == 9;
    return cond1 || cond2;
}

static void rotate_axis2(const npy_bool in[125], npy_bool out[125]) {
    memset(out, 0, 125 * sizeof(npy_bool));
    for (int x = 0; x < 5; x++) {
        for (int y = 0; y < 5; y++) {
            for (int z = 0; z < 5; z++) {
                out[((4 - z) * 5 + y) * 5 + x] = in[(x * 5 + y) * 5 + z];
            }
        }
    }
}

static void rotate_axis3(const npy_bool in[125], npy_bool out[125]) {
    memset(out, 0, 125 * sizeof(npy_bool));
    for (int x = 0; x < 5; x++) {
        for (int y = 0; y < 5; y++) {
            for (int z = 0; z < 5; z++) {
                out[((4 - y) * 5 + x) * 5 + z] = in[(x * 5 + y) * 5 + z];
            }
        }
    }
}

static int shape_preserving_c(npy_bool *data, npy_intp sx, npy_intp sy, npy_intp sz, npy_intp x, npy_intp y, npy_intp z) {
    npy_bool cfg[125];
    npy_bool rot2[125];
    npy_bool rot3[125];
    int idx = 0;
    for (int dx = -2; dx <= 2; dx++) {
        for (int dy = -2; dy <= 2; dy++) {
            for (int dz = -2; dz <= 2; dz++) {
                cfg[idx++] = get3(data, sx, sy, sz, x + dx, y + dy, z + dz);
            }
        }
    }
    if (c12_reference_c(cfg)) return 1;
    rotate_axis2(cfg, rot2);
    if (c12_reference_c(rot2)) return 1;
    rotate_axis3(cfg, rot3);
    if (c12_reference_c(rot3)) return 1;
    return 0;
}

static int count_components_3x3(npy_bool active[27], int use26, int labels_out[27]) {
    int labels[27] = {0};
    int component = 0;
    int queue[27];
    for (int start = 0; start < 27; start++) {
        if (!active[start] || labels[start]) continue;
        component++;
        int qh = 0, qt = 0;
        queue[qt++] = start;
        labels[start] = component;
        while (qh < qt) {
            int item = queue[qh++];
            int ix = item % 3;
            int iy = (item / 3) % 3;
            int iz = item / 9;
            for (int other = 0; other < 27; other++) {
                if (!active[other] || labels[other]) continue;
                int ox = other % 3;
                int oy = (other / 3) % 3;
                int oz = other / 9;
                int dist = abs(ix - ox) + abs(iy - oy) + abs(iz - oz);
                int cheb = abs(ix - ox);
                if (abs(iy - oy) > cheb) cheb = abs(iy - oy);
                if (abs(iz - oz) > cheb) cheb = abs(iz - oz);
                int connected = use26 ? (cheb == 1) : (dist == 1);
                if (connected) {
                    labels[other] = component;
                    queue[qt++] = other;
                }
            }
        }
    }
    if (labels_out != NULL) {
        for (int i = 0; i < 27; i++) labels_out[i] = labels[i];
    }
    return component;
}

static int count_components_2d_8(npy_bool active[9]) {
    int labels[9] = {0};
    int component = 0;
    int queue[9];
    for (int start = 0; start < 9; start++) {
        if (!active[start] || labels[start]) continue;
        component++;
        int qh = 0, qt = 0;
        queue[qt++] = start;
        labels[start] = component;
        while (qh < qt) {
            int item = queue[qh++];
            int ix = item % 3;
            int iy = item / 3;
            for (int other = 0; other < 9; other++) {
                if (!active[other] || labels[other]) continue;
                int ox = other % 3;
                int oy = other / 3;
                int cheb = abs(ix - ox);
                if (abs(iy - oy) > cheb) cheb = abs(iy - oy);
                if (cheb == 1) {
                    labels[other] = component;
                    queue[qt++] = other;
                }
            }
        }
    }
    return component;
}

static int c3_single_orientation_c(const npy_bool cfg[125]) {
    int all_epoints = cfg5_index(cfg, 33) && cfg5_index(cfg, 43) && cfg5_index(cfg, 83) && cfg5_index(cfg, 93);
    npy_bool middle_plane[9];
    int idx = 0;
    for (int y = 1; y <= 3; y++) {
        for (int z = 1; z <= 3; z++) {
            middle_plane[idx++] = cfg[(2 * 5 + y) * 5 + z];
        }
    }
    middle_plane[4] = 0;
    int single_component = count_components_2d_8(middle_plane) == 1;
    int no_tunnel = !(cfg5_index(cfg, 38) && cfg5_index(cfg, 58) && cfg5_index(cfg, 68) && cfg5_index(cfg, 88));
    return (!all_epoints) || (single_component && no_tunnel);
}

static int tunnel_preserving_e_point_c(npy_bool *data, npy_intp sx, npy_intp sy, npy_intp sz, npy_intp x, npy_intp y, npy_intp z) {
    npy_bool cfg[125];
    npy_bool rot2[125];
    npy_bool rot3[125];
    int idx = 0;
    for (int dx = -2; dx <= 2; dx++) {
        for (int dy = -2; dy <= 2; dy++) {
            for (int dz = -2; dz <= 2; dz++) {
                cfg[idx++] = get3(data, sx, sy, sz, x + dx, y + dy, z + dz);
            }
        }
    }
    rotate_axis2(cfg, rot2);
    rotate_axis3(cfg, rot3);
    return c3_single_orientation_c(cfg) && c3_single_orientation_c(rot2) && c3_single_orientation_c(rot3);
}

static void c456_single_orientation_c(const npy_bool cfg[125], int *cond_i, int *mbep, int *mcfp) {
    npy_bool mbep_plane[9];
    npy_bool mcfp_plane[9];
    int idx = 0;
    for (int x = 1; x <= 3; x++) {
        for (int z = 1; z <= 3; z++) {
            mbep_plane[idx++] = cfg[(x * 5 + 2) * 5 + z];
        }
    }
    idx = 0;
    for (int x = 1; x <= 3; x++) {
        for (int y = 1; y <= 3; y++) {
            mcfp_plane[idx++] = cfg[(x * 5 + y) * 5 + 2];
        }
    }
    mbep_plane[4] = 0;
    mcfp_plane[4] = 0;
    *mbep = (count_components_2d_8(mbep_plane) == 1) &&
        ((int)cfg5_index(cfg, 38) + (int)cfg5_index(cfg, 62) + (int)cfg5_index(cfg, 64) + (int)cfg5_index(cfg, 88) != 4);
    *mcfp = (count_components_2d_8(mcfp_plane) == 1) &&
        ((int)cfg5_index(cfg, 58) + (int)cfg5_index(cfg, 62) + (int)cfg5_index(cfg, 64) + (int)cfg5_index(cfg, 68) != 4);
    *cond_i = (!cfg5_index(cfg, 62)) && (!cfg5_index(cfg, 65)) && cfg5_index(cfg, 58) && cfg5_index(cfg, 38) && cfg5_index(cfg, 64);
}

static int final_erosion_point_c(npy_bool *data, npy_intp sx, npy_intp sy, npy_intp sz, npy_intp x, npy_intp y, npy_intp z) {
    npy_bool cfg[125];
    npy_bool rotations[3][125];
    int idx = 0;
    for (int dx = -2; dx <= 2; dx++) {
        for (int dy = -2; dy <= 2; dy++) {
            for (int dz = -2; dz <= 2; dz++) {
                cfg[idx++] = get3(data, sx, sy, sz, x + dx, y + dy, z + dz);
            }
        }
    }
    memcpy(rotations[0], cfg, 125 * sizeof(npy_bool));
    rotate_axis2(cfg, rotations[1]);
    rotate_axis3(cfg, rotations[2]);

    int cond_count = 0;
    int mbep_count = 0;
    int mcfp_count = 0;
    for (int i = 0; i < 3; i++) {
        int cond_i, mbep, mcfp;
        c456_single_orientation_c(rotations[i], &cond_i, &mbep, &mcfp);
        if (cond_i) {
            cond_count++;
            mbep_count += mbep;
            mcfp_count += mcfp;
        }
    }
    if (cond_count == 1) return mcfp_count > 0 && mbep_count > 0;
    if (cond_count == 2) return mcfp_count > 1 || mbep_count > 1;
    if (cond_count == 3) return 1;
    return 0;
}

static int classify_spoints_c(npy_bool cfg[27]) {
    int sp[6] = {4, 10, 12, 16, 14, 22};
    int num = 0;
    for (int i = 0; i < 6; i++) num += cfg[sp[i]] ? 1 : 0;
    int num_opposite = 0;
    if (cfg[sp[0]] && cfg[sp[5]]) num_opposite += 2;
    if (cfg[sp[2]] && cfg[sp[4]]) num_opposite += 2;
    if (cfg[sp[1]] && cfg[sp[3]]) num_opposite += 2;
    int num_adjacent = num - num_opposite;
    if (num == 6) return 0;
    if (num == 5) return 1;
    if (num_opposite == 4) return 2;
    if (num_opposite == 2 && num_adjacent == 2) return 3;
    if (num_opposite == 2 && num_adjacent == 1) return 4;
    if (num_adjacent == 3) return 5;
    if (num_opposite == 2) return 6;
    if (num_adjacent == 2) return 7;
    if (num_adjacent == 1) return 8;
    return 9;
}

static int simple_point_c(npy_bool *data, npy_intp sx, npy_intp sy, npy_intp sz, npy_intp x, npy_intp y, npy_intp z) {
    npy_bool cfg[27];
    int idx = 0;
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                cfg[idx++] = get3(data, sx, sy, sz, x + dx, y + dy, z + dz);
            }
        }
    }
    int class_id = classify_spoints_c(cfg);
    if (class_id == 0) return 0;

    npy_bool black[27];
    for (int i = 0; i < 27; i++) black[i] = cfg[i];
    black[13] = 0;
    int epsilon = count_components_3x3(black, 1, NULL);

    npy_bool background[27];
    for (int i = 0; i < 27; i++) background[i] = !cfg[i];
    background[13] = 0;
    int corners[8] = {0, 6, 2, 8, 18, 20, 24, 26};
    for (int i = 0; i < 8; i++) background[corners[i]] = 0;
    int labels[27];
    count_components_3x3(background, 0, labels);
    int sp[6] = {4, 10, 12, 16, 14, 22};
    int seen[28] = {0};
    int intersecting = 0;
    for (int i = 0; i < 6; i++) {
        int label = labels[sp[i]];
        if (label && !seen[label]) {
            seen[label] = 1;
            intersecting++;
        }
    }
    int mu = intersecting - 1;
    return epsilon == 1 && mu == 0;
}

static int preclassify_c(int epsilon, int mu, int delta) {
    if (epsilon == 0 && mu == 0 && delta == 0) return 1;
    if (epsilon == 1 && mu == 0 && delta == 0) return 2;
    if (epsilon == 2 && mu == 0 && delta == 0) return 3;
    if (epsilon > 2 && mu == 0 && delta == 0) return 4;
    if (epsilon == 1 && mu == 1 && delta == 0) return 5;
    if (epsilon > 1 && mu >= 1 && delta == 0) return 6;
    if (epsilon == 1 && mu > 1 && delta == 0) return 7;
    if (epsilon == 1 && mu == 0 && delta == 1) return 8;
    return 0;
}

static int initial_class_from_key_c(npy_uint32 key, npy_uint8 *out_class) {
    npy_bool cfg[27];
    for (int bit = 0; bit < 27; bit++) {
        cfg[bit] = (key & (((npy_uint32)1) << bit)) ? 1 : 0;
    }

    int class_id = classify_spoints_c(cfg);
    npy_bool black[27];
    for (int i = 0; i < 27; i++) black[i] = cfg[i];
    black[13] = 0;
    int epsilon = count_components_3x3(black, 1, NULL);

    npy_bool background[27];
    for (int i = 0; i < 27; i++) background[i] = !cfg[i];
    background[13] = 0;
    int corners[8] = {0, 6, 2, 8, 18, 20, 24, 26};
    for (int i = 0; i < 8; i++) background[corners[i]] = 0;
    int labels[27];
    count_components_3x3(background, 0, labels);
    int sp[6] = {4, 10, 12, 16, 14, 22};
    int seen[28] = {0};
    int intersecting = 0;
    for (int i = 0; i < 6; i++) {
        int label = labels[sp[i]];
        if (label && !seen[label]) {
            seen[label] = 1;
            intersecting++;
        }
    }
    int mu = intersecting - 1;
    int delta = 0;
    if (class_id == 0) {
        mu = 0;
        delta = 1;
    }

    int initial_class = preclassify_c(epsilon, mu, delta);
    if (initial_class == 0) {
        return 0;
    }
    *out_class = (npy_uint8)initial_class;
    return 1;
}

static PyObject *initial_classes_from_keys(PyObject *self, PyObject *args) {
    PyObject *keys_obj = NULL;
    PyArrayObject *keys = NULL;

    if (!PyArg_ParseTuple(args, "O", &keys_obj)) {
        return NULL;
    }

    keys = (PyArrayObject *)PyArray_FROM_OTF(keys_obj, NPY_UINT32, NPY_ARRAY_IN_ARRAY);
    if (keys == NULL) {
        return NULL;
    }

    PyArrayObject *classes = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(keys), PyArray_DIMS(keys), NPY_UINT8);
    if (classes == NULL) {
        Py_DECREF(keys);
        return NULL;
    }

    npy_intp n = PyArray_SIZE(keys);
    npy_uint32 *key_data = (npy_uint32 *)PyArray_DATA(keys);
    npy_uint8 *class_data = (npy_uint8 *)PyArray_DATA(classes);
    for (npy_intp i = 0; i < n; i++) {
        if (!initial_class_from_key_c(key_data[i], &class_data[i])) {
            PyErr_SetString(PyExc_ValueError, "unclassified topological numbers for neighborhood key");
            Py_DECREF(keys);
            Py_DECREF(classes);
            return NULL;
        }
    }

    Py_DECREF(keys);
    return (PyObject *)classes;
}

static PyObject *propagate_labels_6_connected(PyObject *self, PyObject *args) {
    PyObject *bone_obj = NULL;
    PyObject *labels_obj = NULL;
    PyArrayObject *bone = NULL;
    PyArrayObject *labels = NULL;
    PyArrayObject *out = NULL;
    npy_intp *queue = NULL;

    if (!PyArg_ParseTuple(args, "OO", &bone_obj, &labels_obj)) {
        return NULL;
    }

    bone = (PyArrayObject *)PyArray_FROM_OTF(bone_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
    labels = (PyArrayObject *)PyArray_FROM_OTF(labels_obj, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    if (bone == NULL || labels == NULL) {
        Py_XDECREF(bone);
        Py_XDECREF(labels);
        return NULL;
    }

    if (PyArray_NDIM(bone) != 3 || PyArray_NDIM(labels) != 3) {
        PyErr_SetString(PyExc_ValueError, "bone and seed_labels must be 3D arrays");
        goto fail;
    }
    for (int axis = 0; axis < 3; axis++) {
        if (PyArray_DIM(bone, axis) != PyArray_DIM(labels, axis)) {
            PyErr_SetString(PyExc_ValueError, "bone and seed_labels must have the same shape");
            goto fail;
        }
    }

    out = (PyArrayObject *)PyArray_ZEROS(3, PyArray_DIMS(bone), NPY_UINT8, 0);
    if (out == NULL) {
        goto fail;
    }

    npy_intp sx = PyArray_DIM(bone, 0);
    npy_intp sy = PyArray_DIM(bone, 1);
    npy_intp sz = PyArray_DIM(bone, 2);
    npy_intp total = sx * sy * sz;
    queue = (npy_intp *)malloc((size_t)total * sizeof(npy_intp));
    if (queue == NULL) {
        PyErr_NoMemory();
        goto fail;
    }

    npy_bool *bone_data = (npy_bool *)PyArray_DATA(bone);
    npy_uint8 *label_data = (npy_uint8 *)PyArray_DATA(labels);
    npy_uint8 *out_data = (npy_uint8 *)PyArray_DATA(out);
    npy_intp head = 0;
    npy_intp tail = 0;

    for (npy_intp i = 0; i < total; i++) {
        if (label_data[i] != 0 && bone_data[i]) {
            out_data[i] = label_data[i];
            queue[tail++] = i;
        }
    }

    while (head < tail) {
        npy_intp idx = queue[head++];
        npy_uint8 label = out_data[idx];
        npy_intp x = idx / (sy * sz);
        npy_intp yz = idx - x * sy * sz;
        npy_intp y = yz / sz;
        npy_intp z = yz - y * sz;

        npy_intp neighbors[6];
        int count = 0;
        if (x > 0) neighbors[count++] = idx - sy * sz;
        if (x + 1 < sx) neighbors[count++] = idx + sy * sz;
        if (y > 0) neighbors[count++] = idx - sz;
        if (y + 1 < sy) neighbors[count++] = idx + sz;
        if (z > 0) neighbors[count++] = idx - 1;
        if (z + 1 < sz) neighbors[count++] = idx + 1;

        for (int n = 0; n < count; n++) {
            npy_intp neighbor = neighbors[n];
            if (bone_data[neighbor] && out_data[neighbor] == 0) {
                out_data[neighbor] = label;
                queue[tail++] = neighbor;
            }
        }
    }

    free(queue);
    Py_DECREF(bone);
    Py_DECREF(labels);
    return (PyObject *)out;

fail:
    free(queue);
    Py_XDECREF(out);
    Py_XDECREF(bone);
    Py_XDECREF(labels);
    return NULL;
}

static PyObject *neighborhood_keys_3x3_at(PyObject *self, PyObject *args) {
    PyObject *image_obj = NULL;
    PyObject *coords_obj = NULL;
    PyArrayObject *image = NULL;
    PyArrayObject *coords = NULL;

    if (!PyArg_ParseTuple(args, "OO", &image_obj, &coords_obj)) {
        return NULL;
    }

    image = (PyArrayObject *)PyArray_FROM_OTF(image_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
    coords = (PyArrayObject *)PyArray_FROM_OTF(coords_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    if (image == NULL || coords == NULL) {
        Py_XDECREF(image);
        Py_XDECREF(coords);
        return NULL;
    }

    if (PyArray_NDIM(image) != 3 || PyArray_NDIM(coords) != 2 || PyArray_DIM(coords, 1) != 3) {
        PyErr_SetString(PyExc_ValueError, "image must be 3D and coords must have shape (n, 3)");
        Py_DECREF(image);
        Py_DECREF(coords);
        return NULL;
    }

    npy_intp n = PyArray_DIM(coords, 0);
    npy_intp dims[1] = {n};
    PyArrayObject *keys = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_UINT32);
    if (keys == NULL) {
        Py_DECREF(image);
        Py_DECREF(coords);
        return NULL;
    }

    npy_intp sx = PyArray_DIM(image, 0);
    npy_intp sy = PyArray_DIM(image, 1);
    npy_intp sz = PyArray_DIM(image, 2);
    npy_uint32 *key_data = (npy_uint32 *)PyArray_DATA(keys);
    npy_int64 *coord_data = (npy_int64 *)PyArray_DATA(coords);

    for (npy_intp i = 0; i < n; i++) {
        npy_int64 x = coord_data[i * 3 + 0];
        npy_int64 y = coord_data[i * 3 + 1];
        npy_int64 z = coord_data[i * 3 + 2];
        if (x <= 0 || y <= 0 || z <= 0 || x >= sx - 1 || y >= sy - 1 || z >= sz - 1) {
            PyErr_SetString(PyExc_ValueError, "coords must be at least one voxel inside image bounds");
            Py_DECREF(image);
            Py_DECREF(coords);
            Py_DECREF(keys);
            return NULL;
        }

        npy_uint32 key = 0;
        int bit = 0;
        for (npy_int64 dz = -1; dz <= 1; dz++) {
            for (npy_int64 dy = -1; dy <= 1; dy++) {
                for (npy_int64 dx = -1; dx <= 1; dx++) {
                    npy_bool value = *(npy_bool *)PyArray_GETPTR3(image, x + dx, y + dy, z + dz);
                    if (value) {
                        key |= ((npy_uint32)1) << bit;
                    }
                    bit++;
                }
            }
        }
        key_data[i] = key;
    }

    Py_DECREF(image);
    Py_DECREF(coords);
    return (PyObject *)keys;
}

static PyObject *skeletonize_surface(PyObject *self, PyObject *args) {
    PyObject *image_obj = NULL;
    PyArrayObject *image = NULL;
    PyArrayObject *out = NULL;
    npy_bool *current = NULL;
    npy_bool *start = NULL;
    npy_bool *protected = NULL;
    npy_uint8 *point_types = NULL;
    int max_iterations = 200;

    if (!PyArg_ParseTuple(args, "Oi", &image_obj, &max_iterations)) {
        return NULL;
    }

    image = (PyArrayObject *)PyArray_FROM_OTF(image_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
    if (image == NULL) {
        return NULL;
    }
    if (PyArray_NDIM(image) != 3) {
        PyErr_SetString(PyExc_ValueError, "skeletonize_surface expects a 3D array");
        Py_DECREF(image);
        return NULL;
    }
    if (max_iterations < 0) {
        PyErr_SetString(PyExc_ValueError, "max_iterations must be non-negative");
        Py_DECREF(image);
        return NULL;
    }

    npy_intp sx = PyArray_DIM(image, 0);
    npy_intp sy = PyArray_DIM(image, 1);
    npy_intp sz = PyArray_DIM(image, 2);
    npy_intp psx = sx + 4;
    npy_intp psy = sy + 4;
    npy_intp psz = sz + 4;
    npy_intp padded_size = psx * psy * psz;

    current = (npy_bool *)calloc((size_t)padded_size, sizeof(npy_bool));
    start = (npy_bool *)calloc((size_t)padded_size, sizeof(npy_bool));
    protected = (npy_bool *)calloc((size_t)padded_size, sizeof(npy_bool));
    point_types = (npy_uint8 *)calloc((size_t)padded_size, sizeof(npy_uint8));
    if (current == NULL || start == NULL || protected == NULL || point_types == NULL) {
        PyErr_NoMemory();
        goto fail;
    }

    for (npy_intp x = 0; x < sx; x++) {
        for (npy_intp y = 0; y < sy; y++) {
            for (npy_intp z = 0; z < sz; z++) {
                current[index3(x + 2, y + 2, z + 2, psy, psz)] =
                    *(npy_bool *)PyArray_GETPTR3(image, x, y, z);
            }
        }
    }

    for (int iteration = 0; iteration < max_iterations; iteration++) {
        memcpy(start, current, (size_t)padded_size * sizeof(npy_bool));
        memset(point_types, 0, (size_t)padded_size * sizeof(npy_uint8));
        int candidate_count = 0;

        for (npy_intp x = 2; x < psx - 2; x++) {
            for (npy_intp y = 2; y < psy - 2; y++) {
                for (npy_intp z = 2; z < psz - 2; z++) {
                    npy_intp idx = index3(x, y, z, psy, psz);
                    if (!start[idx] || protected[idx]) {
                        continue;
                    }
                    int neighborhood_count = 0;
                    for (int dx = -1; dx <= 1; dx++) {
                        for (int dy = -1; dy <= 1; dy++) {
                            for (int dz = -1; dz <= 1; dz++) {
                                neighborhood_count += get3(start, psx, psy, psz, x + dx, y + dy, z + dz) ? 1 : 0;
                            }
                        }
                    }
                    if (neighborhood_count >= 27) {
                        continue;
                    }
                    int point_type = border_point_type_c(start, psx, psy, psz, x, y, z);
                    if (point_type > 0) {
                        point_types[idx] = (npy_uint8)point_type;
                        candidate_count++;
                    }
                }
            }
        }

        if (candidate_count == 0) {
            break;
        }

        int removed_this_iteration = 0;
        for (int point_type = 1; point_type <= 3; point_type++) {
            for (int px = 0; px <= 1; px++) {
                for (int py = 0; py <= 1; py++) {
                    for (int pz = 0; pz <= 1; pz++) {
                        for (npy_intp x = 2; x < psx - 2; x++) {
                            if ((x & 1) != px) continue;
                            for (npy_intp y = 2; y < psy - 2; y++) {
                                if ((y & 1) != py) continue;
                                for (npy_intp z = 2; z < psz - 2; z++) {
                                    if ((z & 1) != pz) continue;
                                    npy_intp idx = index3(x, y, z, psy, psz);
                                    if (point_types[idx] != point_type || !current[idx] || protected[idx]) {
                                        continue;
                                    }
                                    if (endpoint_c(current, psx, psy, psz, x, y, z)) {
                                        protected[idx] = 1;
                                        continue;
                                    }
                                    if (shape_preserving_c(start, psx, psy, psz, x, y, z)) {
                                        protected[idx] = 1;
                                        continue;
                                    }
                                    if (point_type == 2 && tunnel_preserving_e_point_c(start, psx, psy, psz, x, y, z)) {
                                        current[idx] = 0;
                                        removed_this_iteration++;
                                    } else if (point_type != 2 && simple_point_c(current, psx, psy, psz, x, y, z)) {
                                        current[idx] = 0;
                                        removed_this_iteration++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if (removed_this_iteration == 0) {
            break;
        }
    }

    memset(point_types, 0, (size_t)padded_size * sizeof(npy_uint8));
    int final_removed = 0;
    for (npy_intp x = 2; x < psx - 2; x++) {
        for (npy_intp y = 2; y < psy - 2; y++) {
            for (npy_intp z = 2; z < psz - 2; z++) {
                npy_intp idx = index3(x, y, z, psy, psz);
                if (!current[idx]) {
                    continue;
                }
                if (final_erosion_point_c(current, psx, psy, psz, x, y, z) && simple_point_c(current, psx, psy, psz, x, y, z)) {
                    point_types[idx] = 1;
                    final_removed++;
                }
            }
        }
    }
    if (final_removed > 0) {
        for (npy_intp i = 0; i < padded_size; i++) {
            if (point_types[i]) {
                current[i] = 0;
            }
        }
    }

    npy_intp dims[3] = {sx, sy, sz};
    out = (PyArrayObject *)PyArray_SimpleNew(3, dims, NPY_BOOL);
    if (out == NULL) {
        goto fail;
    }
    for (npy_intp x = 0; x < sx; x++) {
        for (npy_intp y = 0; y < sy; y++) {
            for (npy_intp z = 0; z < sz; z++) {
                *(npy_bool *)PyArray_GETPTR3(out, x, y, z) =
                    current[index3(x + 2, y + 2, z + 2, psy, psz)];
            }
        }
    }

    free(current);
    free(start);
    free(protected);
    free(point_types);
    Py_DECREF(image);
    return (PyObject *)out;

fail:
    free(current);
    free(start);
    free(protected);
    free(point_types);
    Py_XDECREF(out);
    Py_DECREF(image);
    return NULL;
}

static PyMethodDef methods[] = {
    {"neighborhood_keys_3x3_at", neighborhood_keys_3x3_at, METH_VARARGS, "Pack selected 3x3x3 neighborhoods."},
    {"initial_classes_from_keys", initial_classes_from_keys, METH_VARARGS, "Classify packed 3x3x3 neighborhoods into Saha initial classes."},
    {"propagate_labels_6_connected", propagate_labels_6_connected, METH_VARARGS, "Propagate uint8 seed labels through a 6-connected binary mask."},
    {"skeletonize_surface", skeletonize_surface, METH_VARARGS, "Run topology-preserving surface thinning."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_c_backend",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit__c_backend(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
