#include <array>
#include <cstdint>
#include <functional>
#include <utility>

// #include <string_view>

namespace aie {

// using namespace std::literals::string_view_literals;

// template <typename T, std::size_t Size> using buffer_t = std::array<T, Size>;
template <typename T, std::size_t Size>
struct buffer{
  using storage_t = std::array<T, Size>;
  storage_t storage;
  operator storage_t&() { return storage; }
  // The previous is not enough
  decltype(auto) operator [](std::size_t index) { return storage[index]; }

};

// Typically compiled as:
// !ty_22aie3A3Atile3C12C_43E22 = !cir.struct<struct "aie::tile<1, 4>"
// {!cir.int<u, 8>}>
template <int X, int Y> struct tile {
  static constexpr auto x() { return X; }
  static constexpr auto y() { return Y; }
  template <int X2, int Y2>
  friend constexpr auto operator==(const tile &, const tile<X2, Y2> &) {
    return X == X2 && Y == Y2;
  };
  /*  template <int X1, int Y1, int X2, int Y2>
    friend auto operator<=>(const tile_t<X1, Y1> &, const tile_t<X2, Y2> &) {
      return std::strong_ordering::equal;
    };*/
  /*
  friend constexpr auto operator<=>(const tile &a, const tile &b) {
//    return operator<=>(std::array { a.x, a.y }, std::array { b.x, b.y });
    return operator<=>(a.x, a.y);
  }
*/

  template <typename Code> static inline Code tile_code;

  template <typename T, std::size_t Size> buffer<T, Size> buffer() __attribute__((annotate("aie.tile.buffer"))) {
    return {};
  }

  // Typically compiled as:
  // cir.call @_ZN3aie4tileILi1ELi4EE7programIZ4mainE3$_0EEvOT_(%2, %7) :
  // (!cir.ptr<!ty_22aie3A3Atile3C12C_43E22>, !cir.ptr<!ty_22anon2E122>) -> ()
  // loc(#loc63)
  void program(auto &&code) __attribute__((annotate("aie.tile.program"))) {
    // Use this since std::function crashes ClangIR 2024/09/12
    //tile_code<decltype(&code)> = &code;
    // Just to instantiate the lambda body while waiting for std::function
    code();
  }
};

//template <int X, int Y> inline constexpr tile<X, Y> tile;

template <typename Storage> struct tile_handle {
  Storage tile_memory;
  constexpr tile_handle(Storage tile_memory) : tile_memory{tile_memory} {};
  constexpr auto &mem() { return tile_memory; }
};

// Inject in aie:: to ease use
enum : std::int8_t { npu1 = 42};

// Typically compiled as:
// !ty_22aie3A3Adevice3Caie3A3Anpu13E22 = !cir.struct<struct
// "aie::device<aie::npu1>" {!cir.int<u, 8>}>
template <auto Name = npu1> struct device {
  template <int X, int Y>
  tile<X, Y> tile()
      __attribute__((annotate("aie.device.tile", X, Y, Name, std::to_underlying(Name)))) {
    return {};
  };

  void constexpr run() {}
};

template <typename T> struct tile_storage_t {
  using content = T;
};

template <typename T> inline constexpr tile_storage_t<T> tile_storage;

} // namespace aie
