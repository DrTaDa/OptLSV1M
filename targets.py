import numpy
import gc

from mozaik.storage.queries import param_filter_query
from utils import *
from math import isclose
from scipy.spatial.distance import cdist


class TargetValue():

    def __init__(self, name, target_value, sheet_name=None, norm=None, max_score=10):
        self.name  = name
        self.target_value = target_value
        self.norm = norm
        self.max_score = max_score
        self.sheet_name = sheet_name
        assert self.norm != 0

    def calculate_value(self, data_store):
        raise NotImplementedError

    def calculate_score(self, data_store):
        if data_store is None:
            return self.max_score
        value = self.calculate_value(data_store)
        if value is None or numpy.isnan(value):
            score = self.max_score
        elif self.norm is not None:
            score = abs(self.target_value - value) / self.norm
        else:
            score = abs(self.target_value - value)
        threshold_score = min(score, self.max_score)
        if numpy.isnan(threshold_score):
            threshold_score = self.max_score
        if value is None:
            print("For feature {}, value is None, score set to {:.2f}".format(self.name, threshold_score))
        else:
            print("For feature {}, for value {:.4f} computed score {:.2f}".format(self.name, value, threshold_score))
        return threshold_score


class OneBoundUpperTarget(TargetValue):
    def calculate_score(self, data_store):
        if data_store is None:
            return self.max_score
        value = self.calculate_value(data_store)
        if value is None or numpy.isnan(value):
            score = self.max_score
        elif self.norm is not None:
            if value < self.target_value:
                score = abs(self.target_value - value) / self.norm
            else:
                score = 0.
        else:
            if value < self.target_value:
                score = abs(self.target_value - value)
            else:
                score = 0.
        threshold_score = min(score, self.max_score)
        if numpy.isnan(threshold_score):
            threshold_score = self.max_score
        print("For feature {}, for value {:.4f} computed score {:.2f}".format(self.name, value, threshold_score))
        return threshold_score


class OneBoundLowerTarget(TargetValue):
    def calculate_score(self, data_store):
        if data_store is None:
            return self.max_score
        value = self.calculate_value(data_store)
        if value is None or numpy.isnan(value):
            score = self.max_score
        elif self.norm is not None:
            if value > self.target_value:
                score = abs(self.target_value - value) / self.norm
            else:
                score = 0.
        else:
            if value > self.target_value:
                score = abs(self.target_value - value)
            else:
                score = 0.
        threshold_score = min(score, self.max_score)
        if numpy.isnan(threshold_score):
            threshold_score = self.max_score
        print("For feature {}, for value {:.4f} computed score {:.2f}".format(self.name, value, threshold_score))
        return threshold_score


class SpontActivityTarget(TargetValue):
    def calculate_value(self, data_store):

        segments = param_filter_query(data_store, sheet_name=self.sheet_name, st_name='InternalStimulus').get_segments()
        if not len(segments):
            return None

        spiketrains = segments[0].spiketrains
        return 1000 * numpy.mean([float(len(s) / (s.t_stop - s.t_start)) for s in spiketrains])


class IrregularityTarget(OneBoundUpperTarget):
    def calculate_value(self, data_store):
        
        segments = param_filter_query(data_store, sheet_name=self.sheet_name, st_name='InternalStimulus').get_segments()
        if not len(segments):
            return None

        spiketrains = segments[0].spiketrains
        isis = [numpy.diff(st.magnitude) for st in spiketrains]
        idxs = numpy.array([len(isi) for isi in isis]) > 5
        value = numpy.mean(numpy.array([numpy.std(isi) / numpy.mean(isi) for isi in isis])[idxs])
        return value


class SynchronyTarget(TargetValue):
    def calculate_value(self, data_store):
        gc.collect()

        segments = param_filter_query(data_store, sheet_name=self.sheet_name, st_name='InternalStimulus').get_segments()
        if not len(segments):
            return None

        spiketrains = segments[0].spiketrains[:2000]
        isis = [numpy.diff(st.magnitude) for st in spiketrains]
        idxs = numpy.array([len(isi) for isi in isis]) > 5
        t_start = round(spiketrains[0].t_start, 5)
        t_stop = round(spiketrains[0].t_stop, 5)
        num_bins = int(round((t_stop - t_start) / 10.))
        r = (float(t_start), float(t_stop))
        psths = [numpy.histogram(x, bins=num_bins, range=r)[0] for x in spiketrains]

        corrs = numpy.nan_to_num(numpy.corrcoef(numpy.squeeze(psths)))
        value = numpy.mean(corrs[idxs, :][:, idxs][numpy.triu_indices(sum(idxs == True), 1)])

        gc.collect()

        return value


class OrientationTuningPreferenceTarget(OneBoundUpperTarget):
    def calculate_value(self, data_store):
        gc.collect()
        # Choose the O to consider
        target_O = numpy.pi / 2
        orthogonal_O = target_O - (numpy.pi / 2)

        # Filter the cell based on what has been recorded and the target orientation
        segments = param_filter_query(data_store, sheet_name=self.sheet_name, st_name='InternalStimulus').get_segments()
        if not len(segments):
            return None

        seg = segments[0]
        orientations_cells = get_orientation_preference(data_store, self.sheet_name).values
        idxs = data_store.get_sheet_indexes(self.sheet_name, seg.get_stored_spike_train_ids())
        idx_cells = [idx for idx in idxs if isclose(target_O, orientations_cells[idx], abs_tol=0.1)]

        # Get the id of the neurons that respond to that O
        orientations_cells = get_orientation_preference(data_store, self.sheet_name).values
        idx_cells = [idx for idx, oc in enumerate(orientations_cells) if isclose(target_O, oc, abs_tol=0.1)]

        # Get the rates for the O and the orthogonal
        segments = param_filter_query(data_store, st_name='FullfieldDriftingSinusoidalGrating', sheet_name=self.sheet_name, st_contrast=[100]).get_segments()
        if not len(segments):
            return None

        for seg in segments:
            orientation = get_annotation(seg, "orientation")
            if isclose(orientation, target_O, abs_tol=0.1):
                spike_rate_high = get_mean_rate(seg.spiketrains, idx_cells)
            elif isclose(orientation, orthogonal_O, abs_tol=0.1):
                spike_rate_ortho_high = get_mean_rate(seg.spiketrains, idx_cells)

        gc.collect()

        return spike_rate_high / spike_rate_ortho_high


class OrientationTuningOrthoHighTarget(TargetValue):
    def calculate_value(self, data_store):
        gc.collect()

        # Choose the O to consider
        target_O = numpy.pi / 2
        orthogonal_O = target_O - (numpy.pi / 2)

        # Filter the cell based on what has been recorded and the target orientation
        segments = param_filter_query(data_store, sheet_name=self.sheet_name, st_name='InternalStimulus').get_segments()
        if not len(segments):
            return None
        seg = segments[0]
        orientations_cells = get_orientation_preference(data_store, self.sheet_name).values
        idxs = data_store.get_sheet_indexes(self.sheet_name, seg.get_stored_spike_train_ids())
        idx_cells = [idx for idx in idxs if isclose(target_O, orientations_cells[idx], abs_tol=0.1)]

        # Get the spontaneous rate
        spont_rate = get_mean_rate(seg.spiketrains, idx_cells)

        # Get the rates for the O and the orthogonal O high contrast
        segments = param_filter_query(data_store, st_name='FullfieldDriftingSinusoidalGrating', sheet_name=self.sheet_name, st_contrast=[100]).get_segments()
        if not len(segments):
            return None
        for seg in segments:
            orientation = get_annotation(seg, "orientation")
            if isclose(orientation, orthogonal_O, abs_tol=0.1):
                spike_rate_ortho_high = get_mean_rate(seg.spiketrains, idx_cells)
                break

        gc.collect()

        return numpy.abs(spont_rate - spike_rate_ortho_high)  / spont_rate


class OrientationTuningOrthoLowTarget(TargetValue):
    def calculate_value(self, data_store):
        gc.collect()
    
        # Choose the O to consider
        target_O = numpy.pi / 2
        orthogonal_O = target_O - (numpy.pi / 2)
        
        # Filter the cell based on what has been recorded and the target orientation
        segments = param_filter_query(data_store, sheet_name=self.sheet_name, st_name='InternalStimulus').get_segments()
        if not len(segments):
            return None
        seg = segments[0]
        orientations_cells = get_orientation_preference(data_store, self.sheet_name).values
        idxs = data_store.get_sheet_indexes(self.sheet_name, seg.get_stored_spike_train_ids())
        idx_cells = [idx for idx in idxs if isclose(target_O, orientations_cells[idx], abs_tol=0.1)]

        # Get the spontaneous rate
        segments = param_filter_query(data_store, sheet_name=self.sheet_name, st_name='InternalStimulus').get_segments()
        if not len(segments):
            return None
        seg = segments[0]
        spont_rate = get_mean_rate(seg.spiketrains, idx_cells)

        # Get the rates for the orthogonal O low contrast
        segments = param_filter_query(data_store, st_name='FullfieldDriftingSinusoidalGrating', sheet_name=self.sheet_name, st_contrast=[10]).get_segments()
        if not len(segments):
            return None
        for seg in segments:
            orientation = get_annotation(seg, "orientation")
            if isclose(orientation, orthogonal_O, abs_tol=0.1):
                spike_rate_ortho_low = get_mean_rate(seg.spiketrains, idx_cells)
                break

        gc.collect()

        return numpy.abs(spont_rate - spike_rate_ortho_low) / spont_rate


class SizeTuning(TargetValue):
    def calculate_value(self, data_store):
        """Compute the percentage of cells that show a suppression above +20%"""

        gc.collect()

        # Choose the O to consider
        target_O = numpy.pi / 2

        # Filter the cell based on what has been recorded and the target orientation
        segments = param_filter_query(data_store, sheet_name=self.sheet_name, st_name='InternalStimulus').get_segments()
        if not len(segments):
            return None
        seg = segments[0]
        orientations_cells = get_orientation_preference(data_store, self.sheet_name).values
        idxs = data_store.get_sheet_indexes(self.sheet_name, seg.get_stored_spike_train_ids())
        idx_cells = [idx for idx in idxs if isclose(target_O, orientations_cells[idx], abs_tol=0.1)]

        # Get the firing rates for the full field gratings at high contrast
        segments = param_filter_query(
            data_store, st_name='FullfieldDriftingSinusoidalGrating', sheet_name=self.sheet_name, st_contrast=[100]
        ).get_segments()
        if not len(segments):
            return None
        for seg in segments:
            orientation = get_annotation(seg, "orientation")
            if isclose(orientation, target_O, abs_tol=0.1):
                full_field_rates = [len(seg.spiketrains[idx]) for idx in idx_cells]
                break

        # Get the firing rates for the small grating disk at high contrast
        segments = param_filter_query(
            data_store, st_name='DriftingSinusoidalGratingDisk', sheet_name=self.sheet_name, st_contrast=[100]
        ).get_segments()
        if not len(segments):
            return None
        for seg in segments:
            orientation = get_annotation(seg, "orientation")
            if isclose(orientation, target_O, abs_tol=0.1):
                small_disk_rates = [len(seg.spiketrains[idx]) for idx in idx_cells]
                break

        gc.collect()

        suppression_index = []
        for small_rate, full_rate in zip(small_disk_rates, full_field_rates):
            if small_rate > 2. and full_rate > 0.:
                _suppression = small_rate / full_rate
                if _suppression > 1.1:
                    suppression_index.append(_suppression)

        return len(suppression_index) / len(idx_cells)

    
class ICMSTarget():

    def __init__(self, name, target_values, value_distances, sheet_names, active_electrode, stimulation_frequency, amplitude, norms=None, max_score=10):
        self.name  = name
        self.value_distances = value_distances
        self.target_values = target_values
        self.norms = norms
        self.max_score = max_score
        self.sheet_names = sheet_names
        self.active_electrode = active_electrode
        self.stimulation_frequency = stimulation_frequency
        self.n_electrodes = 100
        self.amplitude = amplitude
        assert all(n != 0 for n in self.norms)

    def _get_activity_per_electrode(self, data_store):

        activity_per_electrode = [[] for i in range(self.n_electrodes)]
        selection_radius = 200.
        stimulation_ISI = 1000. / self.stimulation_frequency

        for sheet_name in self.sheet_names:

            data_spike_ICMS = param_filter_query(
                data_store, sheet_name=sheet_name, st_direct_stimulation_name="IntraCorticalMicroStimulation"
            ).get_segments()[0].spiketrains

            probe_positions, cells_positions, _ = get_ICMS_position_data(
                sheet_name, data_store, amplitude=self.amplitude, frequency=self.stimulation_frequency, active_electrodes=[self.active_electrode]
            )
            distances = cdist(probe_positions.values, cells_positions)

            n_neurons = 0
            for electrode_idx in range(self.n_electrodes):
                for cell_idx in range(len(cells_positions)):
                    if distances[electrode_idx, cell_idx] < selection_radius:
                        _spikes = numpy.array(data_spike_ICMS[cell_idx])
                        activity_per_electrode[electrode_idx] += list(
                            ((_spikes[_spikes > stimulation_ISI]) % stimulation_ISI)
                        )
                        n_neurons += 1

            del data_spike_ICMS
            del distances
            gc.collect()

        return activity_per_electrode

    def _order_by_exp_distances(self, values_sim, distances_sim):
        ordered_values = [[] for i in range(len(self.value_distances))]
        for idx_exp, dist_exp in enumerate(self.value_distances):
            for idx_sim, dist_sim in enumerate(distances_sim):
                if numpy.isclose(dist_exp, dist_sim, atol=1):
                    ordered_values[idx_exp].append(values_sim[idx_sim])
        ordered_values = [numpy.mean(ov) if ov else None for ov in ordered_values]
        return ordered_values

    def calculate_values(self, data_store):
        raise NotImplementedError

    def calculate_score(self, data_store):
        if data_store is None:
            return self.max_score
        values = self.calculate_values(data_store)
        #print(values)
        if values is None:
            scores = self.max_score
        else:
            scores = []
            for exp_value, norm, sim_value, dist in zip(self.target_values, self.norms, values, self.value_distances):
                if dist < 2000:
                    if sim_value is None:
                        scores.append(self.max_score)
                    else:
                        #print(exp_value, sim_value, norm, abs(exp_value - sim_value) / norm)
                        scores.append(abs(exp_value - sim_value) / norm)
                        #plt.scatter(dist, exp_value, color="C0")
                        #plt.scatter(dist, sim_value, color="C1")

            score = numpy.sum(scores)
            #plt.show()
            #plt.clf()
        #print(score)
        threshold_score = min(score, self.max_score)
        if numpy.isnan(threshold_score):
            raise Exception("SCORE IS NAN: TO DEBUG")
        return threshold_score


class ICMSExcAmplitudeTarget(ICMSTarget):
    def calculate_values(self, data_store):

        activity_per_electrode = self._get_activity_per_electrode(data_store)

        probe_positions, _, _ = get_ICMS_position_data(
            self.sheet_names[0], data_store, amplitude=self.amplitude, frequency=self.stimulation_frequency, active_electrodes=[self.active_electrode]
        )
        pair_distances = cdist(probe_positions.values[:, :2], probe_positions.values[:, :2])

        excitation_amplitudes = []
        distances = []

        bin_with = 20 # ms
        spont_skip = 1000 # ms
        stimulation_ISI = 1000. / self.stimulation_frequency
        nbins = round(stimulation_ISI / bin_with)
        nbins_spont_skip = round((stimulation_ISI - spont_skip) / bin_with)

        # Compute the values
        for electrode_idx in range(self.n_electrodes):
            hist_raw, bin_edges = numpy.histogram(activity_per_electrode[electrode_idx], bins=nbins, range=(0, stimulation_ISI))
            mean_rate = numpy.mean(hist_raw[nbins_spont_skip:])
            hist_normed = (hist_raw - mean_rate) / mean_rate
            excitation_amplitudes.append(numpy.max(hist_normed[:2]))
            distances.append(pair_distances[self.active_electrode, electrode_idx])

            """if electrode_idx % 10 == 0:
                print(electrode_idx, excitation_amplitudes[-1])
                plt.stairs(hist_raw, bin_edges)
                plt.show()
                plt.clf()"""

        gc.collect()
        # Order by distances of the exp data
        return self._order_by_exp_distances(excitation_amplitudes, distances)


class ICMSInhDurationTarget(ICMSTarget):
    def calculate_values(self, data_store):

        activity_per_electrode = self._get_activity_per_electrode(data_store)

        probe_positions, _, electrode_active_per_cell = get_ICMS_position_data(
            self.sheet_names[0], data_store, amplitude=self.amplitude, frequency=self.stimulation_frequency, active_electrodes=[self.active_electrode]
        )
        pair_distances = cdist(probe_positions.values[:, :2], probe_positions.values[:, :2])

        inhibition_durations = []
        distances = []

        bin_with = 20 # ms
        spont_skip = 1000 # ms
        stimulation_ISI = 1000. / self.stimulation_frequency
        nbins = round(stimulation_ISI / bin_with)
        nbins_spont_skip = round((stimulation_ISI - spont_skip) / bin_with)

        # Compute the values
        for electrode_idx in range(self.n_electrodes):
            hist_raw, bin_edges = numpy.histogram(activity_per_electrode[electrode_idx], bins=nbins, range=(0, stimulation_ISI))
            mean_rate = numpy.mean(hist_raw[nbins_spont_skip:])
            hist_normed = (hist_raw - mean_rate) / mean_rate

            # Compute the length of the inhibitory period
            tmp_smoothed = numpy.convolve(hist_normed, numpy.ones(3) / 3, mode='valid')
            hist_smooth = numpy.copy(hist_normed)
            hist_smooth[1:-1] = tmp_smoothed
        
            inh_start_bin = next((bin_idx for bin_idx in range(4) if hist_smooth[bin_idx] < 0.1), None)
        
            if inh_start_bin is not None:
                inh_end_bin = inh_start_bin + 1
                while inh_end_bin < len(hist_smooth) and hist_smooth[inh_end_bin] < 0.1: inh_end_bin += 1
                inhibition_durations.append((inh_end_bin - inh_start_bin) * bin_with)
            else:
                inhibition_durations.append(0.)

            distances.append(pair_distances[self.active_electrode, electrode_idx])

        gc.collect()
        # Order by distances of the exp data
        return self._order_by_exp_distances(inhibition_durations, distances)


class ICMSInhDepthTarget(ICMSTarget):
    def calculate_values(self, data_store):

        activity_per_electrode = self._get_activity_per_electrode(data_store)

        probe_positions, _, electrode_active_per_cell = get_ICMS_position_data(
            self.sheet_names[0], data_store, amplitude=self.amplitude, frequency=self.stimulation_frequency, active_electrodes=[self.active_electrode]
        )
        pair_distances = cdist(probe_positions.values[:, :2], probe_positions.values[:, :2])

        inhibition_depths = []
        distances = []

        bin_with = 20 # ms
        spont_skip = 1000 # ms
        stimulation_ISI = 1000. / self.stimulation_frequency
        nbins = round(stimulation_ISI / bin_with)
        nbins_spont_skip = round((stimulation_ISI - spont_skip) / bin_with)

        # Compute the values
        for electrode_idx in range(self.n_electrodes):
            hist_raw, bin_edges = numpy.histogram(activity_per_electrode[electrode_idx], bins=nbins, range=(0, stimulation_ISI))
            mean_rate = numpy.mean(hist_raw[nbins_spont_skip:])
            hist_normed = (hist_raw - mean_rate) / mean_rate

            # Compute the length of the inhibitory period
            tmp_smoothed = numpy.convolve(hist_normed, numpy.ones(3) / 3, mode='valid')
            hist_smooth = numpy.copy(hist_normed)
            hist_smooth[1:-1] = tmp_smoothed

            inh_start_bin = next((bin_idx for bin_idx in range(4) if hist_smooth[bin_idx] < 0.1), None)

            if inh_start_bin is not None:
                inh_end_bin = inh_start_bin + 1
                while inh_end_bin < len(hist_smooth) and hist_smooth[inh_end_bin] < 0.1: inh_end_bin += 1
                inhibition_depths.append(numpy.min(hist_smooth[inh_start_bin:inh_end_bin + 1]))
            else:
                inhibition_depths.append(0.)

            distances.append(pair_distances[self.active_electrode, electrode_idx])

        gc.collect()
        # Order by distances of the exp data
        return self._order_by_exp_distances(inhibition_depths, distances)
