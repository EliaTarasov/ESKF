#ifndef RINGBUFFER_H_
#define RINGBUFFER_H_

#include <cstdlib>

namespace eskf {

  template <typename data_type>
  class RingBuffer {
  public:
    RingBuffer() {
      _buffer = NULL;
      _head = _tail = _size = 0;
      _first_write = true;
    }
    ~RingBuffer() { delete[] _buffer; }

    bool allocate(int size) {
      if (size <= 0) {
	return false;
      }

      if (_buffer != NULL) {
	delete[] _buffer;
      }

      _buffer = new data_type[size];

      if (_buffer == NULL) {
	return false;
      }

      _size = size;
      
      // set the time elements to zero so that bad data is not retrieved from the buffers
      for (unsigned index = 0; index < _size; index++) {
	_buffer[index].time_us = 0;
      }
      _first_write = true;
      return true;
    }

    void unallocate() {
      if (_buffer != NULL) {
	delete[] _buffer;
      }
    }

    inline void push(data_type sample) {
      int head_new = _head;

      if (_first_write) {
	head_new = _head;
      } else {
	head_new = (_head + 1) % _size;
      }

      _buffer[head_new] = sample;
      _head = head_new;

      // move tail if we overwrite it
      if (_head == _tail && !_first_write) {
	_tail = (_tail + 1) % _size;
      } else {
	_first_write = false;
      }
    }

    inline data_type get_oldest() {
      return _buffer[_tail];
    }

    unsigned get_oldest_index() {
      return _tail;
    }

    inline data_type get_newest() {
      return _buffer[_head];
    }

    inline bool pop_first_older_than(uint64_t timestamp, data_type *sample) {
      // start looking from newest observation data
      for (unsigned i = 0; i < _size; i++) {
	int index = (_head - i);
	index = index < 0 ? _size + index : index;
	if (timestamp >= _buffer[index].time_us && timestamp - _buffer[index].time_us < 100000) {
	  // TODO Re-evaluate the static cast and usage patterns
	  memcpy(static_cast<void *>(sample), static_cast<void *>(&_buffer[index]), sizeof(*sample));
	  // Now we can set the tail to the item which comes after the one we removed
	  // since we don't want to have any older data in the buffer
	  if (index == static_cast<int>(_head)) {
	    _tail = _head;
	    _first_write = true;
	  } else {
	    _tail = (index + 1) % _size;
	  }
	_buffer[index].time_us = 0;
	return true;
      }
      if (index == static_cast<int>(_tail)) {
	// we have reached the tail and haven't got a match
	return false;
      }
    }
    return false;
  }
  
  data_type &operator[](unsigned index) {
    return _buffer[index];
  }

  // return data at the specified index
  inline data_type get_from_index(unsigned index) {
    if (index >= _size) {
      index = _size-1;
    }
    return _buffer[index];
  }

  // push data to the specified index
  inline void push_to_index(unsigned index, data_type sample) {
    if (index >= _size) {
      index = _size-1;
    }
    _buffer[index] = sample;
  }

  // return the length of the buffer
  unsigned get_length() {
    return _size;
  }

  private:
    data_type *_buffer;
    unsigned _head, _tail, _size;
    bool _first_write;
  };
}

#endif /* defined(RINGBUFFER_H_) */