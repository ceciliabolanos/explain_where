import numpy as np
import warnings

def default_composition_fn(x):
    return x

class Factorization(object):
    def __init__(self):
        pass

    def compose_model_input(self, components=None):
        raise NotImplementedError

    def get_number_components(self):
        raise NotImplementedError

    def retrieve_components(self, selection_order=None):
        raise NotImplementedError

    def get_ordered_component_names(self): # e.g. instrument names
        raise NotImplementedError

    def set_analysis_window(self, start_sample, y_length):
        raise NotImplementedError
    
class DataBasedFactorization(Factorization):

    def __init__(self, data_provider, n_temporal_segments, composition_fn=None):
        """
        :param data_provider: object of class DataProvider
        :param n_temporal_segments: number of temporal segments used in the segmentation
        :param composition_fn: allows to apply transformations to the summed sources,
                e.g. return a spectrogram
                (same factorization class can be used independent of the input the model requires)
        """
        super().__init__()
        if composition_fn is None:
            composition_fn = default_composition_fn
        self.data_provider = data_provider
        self.composition_fn = composition_fn
        self.n_temporal_segments = n_temporal_segments
        self.original_components = []
        self.components = []
        self._components_names = []

        self.initialize_components()  # that's the part that's specific to each source sep. algorithm
        self.set_analysis_window(0, len(self.data_provider.get_mix()))

    def compose_model_input(self, components=None):
        sel_sources = self.retrieve_components(selection_order=components)
        if len(sel_sources) > 1:
            y = sum(sel_sources)
        else:
            y = sel_sources[0]
        return self.composition_fn(y)

    def get_number_components(self):
        return len(self.components)

    def retrieve_components(self, selection_order=None):
        if selection_order is None:
            return self.components
        return [self.components[o] for o in selection_order]

    def get_ordered_component_names(self):
        if len(self._components_names) == 0:
            raise Exception("Components were not named.")
        return self._components_names

    def initialize_components(self):
        raise NotImplementedError

    def prepare_components(self, start_sample, y_length):
        # this resets in case temporal segmentation was previously applied
        self.components = [
            comp[start_sample:start_sample + y_length] for comp in self.original_components]

        mix = self.data_provider.get_mix()
        audio_length = len(mix)
        n_temporal_segments = self.n_temporal_segments
        samples_per_segment = audio_length // n_temporal_segments

        explained_length = samples_per_segment * n_temporal_segments
        if explained_length < audio_length:
            warnings.warn("last {} samples are ignored".format(audio_length - explained_length))

        component_names = []
        temporary_components = []
        for s in range(n_temporal_segments):
            segment_start = s * samples_per_segment
            segment_end = segment_start + samples_per_segment
            for co in range(self.get_number_components()):
                current_component = np.zeros(explained_length, dtype=np.float32)
                current_component[segment_start:segment_end] = self.components[co][segment_start:segment_end]
                temporary_components.append(current_component)
                component_names.append(self._components_names[co]+str(s))

        self.components = temporary_components
        self._components_names = component_names

    def set_analysis_window(self, start_sample, y_length):
        self.data_provider.set_analysis_window(start_sample, y_length)
        self.prepare_components(start_sample, y_length)


class TemporalSegmentationFactorization(Factorization):
    def __init__(self, data_provider, wav, segment_duration, window_duration, sr=16000, composition_fn=None):
        super().__init__()
        if composition_fn is None:
            composition_fn = default_composition_fn
        self.data_provider = data_provider
        self.wav_array = wav
        self.composition_fn = composition_fn
        self.segment_duration_ms = segment_duration
        self.window_duration_ms = window_duration
        self.sr = sr
        self.components = []
        self._components_names = []
        self.initialize_components()
    
    def initialize_components(self):
        # Calculate the number of samples per segment and window
        samples_per_segment = int(self.segment_duration_ms * self.sr / 1000)
        
        # Calculate the total number of samples and segments
        total_samples = len(self.wav_array)
        n_segments = total_samples // samples_per_segment

        # Create temporal segments with sliding window
        for i in range(n_segments):
            start = i * samples_per_segment
            end = start + samples_per_segment
            
            segment = np.zeros_like(self.wav_array)
            segment[start:end] = self.wav_array[start:end]
            
            self.components.append(segment)
            start_time_ms = i * self.window_duration_ms
            end_time_ms = start_time_ms + self.segment_duration_ms
            self._components_names.append(f"Segment_{start_time_ms:.0f}ms_to_{end_time_ms:.0f}ms")

        if end < total_samples:
            start = end
            end = total_samples
            
            segment = np.zeros_like(self.wav_array)
            segment[start:end] = self.wav_array[start:end]
            
            self.components.append(segment)
            start_time_ms = (n_segments) * self.window_duration_ms
            end_time_ms = start_time_ms + ((end - start) / self.sr * 1000)
            self._components_names.append(f"Segment_{start_time_ms:.0f}ms_to_{end_time_ms:.0f}ms")
    
    def get_number_components(self):
        return len(self.components)

    def set_analysis_window(self, start_sample, y_length):
        # This method is simplified as we're working with pre-segmented data
        # You might want to adjust this if you need to change the analysis window dynamically
        pass

    def retrieve_components(self, selection_order=None):
        if selection_order is None:
            return self.components
        return [self.components[o] for o in selection_order]
    
    def compose_model_input(self, components=None):
        sel_sources = self.retrieve_components(selection_order=components)
        if len(sel_sources) > 1:
            y = sum(sel_sources)
        else:
            y = sel_sources[0]
        return self.composition_fn(y)