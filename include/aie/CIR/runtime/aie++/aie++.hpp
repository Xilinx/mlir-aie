#include <array>
#include <cstdint>
#include <functional>
#include <utility>

// #include <string_view>

namespace aie {

// using namespace std::literals::string_view_literals;

// template <typename T, std::size_t Size> using buffer_t = std::array<T, Size>;
template <typename T, std::size_t Size> struct buffer {
  //using storage_t = std::array<T, Size>;
  using storage_t = T[Size];
  storage_t storage;
  operator storage_t&() { return storage; }
  // The previous is not enough
  decltype(auto) operator[](std::size_t index) { return storage[index]; }

  //auto begin() { return storage.begin(); }
  auto begin() { return &storage[0]; }

  //auto end() { return storage.end(); }
  auto end() { return &storage[Size]; }
};

template <typename Channel> void aquire() {}

template <typename Channel> void release() {}

template <typename Channel> struct accessor {
  // clang++:
  // /home/rkeryell/Xilinx/Projects/LLVM/worktrees/clangir/clang/lib/CIR/CodeGen/CIRGenExpr.cpp:2548:
  // LValue cir::CIRGenFunction::buildLValue(const Expr *): Assertion `0 && "not
  // implemented"' failed.
  //
  // Channel& channel;
  //
  // accessor(Channel& c) : channel { c } {aquire<Channel>(); }
  using access_t = typename Channel::access_t;
  access_t* storage;

  // clang++:
  // /home/rkeryell/Xilinx/Projects/LLVM/worktrees/clangir/clang/lib/CIR/CodeGen/CIRGenExpr.cpp:2548:
  // LValue cir::CIRGenFunction::buildLValue(const Expr *): Assertion `0 && "not
  // implemented"' failed.
  //
  // accessor(access_t& a) : storage {a} { aquire<Channel>(); }
  accessor(access_t& a) {
    storage = &a;
    aquire<Channel>();
  }
  auto& operator[](std::size_t index) { return (*storage)[index]; }
  operator typename Channel::access_t &() { return *storage; }
  ~accessor() { release<Channel>(); }
};

// Channel abstraction used to send data between tiles.
//
// This is lowered to MLIR aie.objectfifo.
template <typename ValueType, std::size_t Size, typename FromTile,
          typename ToTile>
struct channel {
  static_assert(std::is_same_v<typename FromTile::device_type,
                               typename ToTile::device_type>,
                "Only tiles from the same device can be connected");
  using from_tile_type = FromTile;
  using to_tile_type = ToTile;
  using device_type = typename to_tile_type::device_type;
  // clang++:
  // /home/rkeryell/Xilinx/Projects/LLVM/worktrees/clangir/clang/lib/CIR/CodeGen/CIRGenExpr.cpp:552:
  // CIRGenCallee cir::CIRGenFunction::buildCallee(const clang::Expr *):
  // Assertion `!dyn_cast<CXXPseudoDestructorExpr>(E) && "NYI"' failed.
  // std::vector<ValueType> storage;
  ValueType storage[10 * Size];
  std::size_t capacity;
  using access_t = ValueType[][Size];
  channel(FromTile& from_tile, ToTile& to_tile, std::size_t capacity)
      : capacity { capacity } {
    // storage.assign(0, capacity * Size);
  }

  auto& get_storage() {
    // \todo
    return *reinterpret_cast<access_t*>(storage);
  }

  auto in_acquire_release(std::size_t capacity) {
    // \todo
    return accessor<channel>(get_storage());
  }

  auto out_acquire_release(std::size_t capacity) {
    // \todo
    return accessor<channel>(get_storage());
  }
};

// Represent the tile in AIE Device.
//
// X is the column number starting at 0.
//
// Y is the row number starting at 0.
//
// Typically compiled as:
// !ty_22aie3A3Atile3C12C_43E22 = !cir.struct<struct "aie::tile<1, 4>"
// {!cir.int<u, 8>}>
//
// The tile depends on a Device since we can have programs with different
// devices at the same time
template <int X, int Y, typename Device> struct tile {
  using device_type = Device;
  static constexpr auto x() { return X; }
  static constexpr auto y() { return Y; }
  // Only tiles from a same device are comparable
  template <int X2, int Y2>
  friend constexpr auto operator==(const tile&, const tile<X2, Y2, Device>&) {
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

  // Get a buffer with Size elements of type T.
  template <typename T, std::size_t Size>
  buffer<T, Size> buffer() __attribute__((annotate("aie.tile.buffer"))) {
    return {};
  }

  // Define the code to be run inside a tile.
  //
  // Typically compiled as:
  // cir.call @_ZN3aie4tileILi1ELi4EE7programIZ4mainE3$_0EEvOT_(%2, %7) :
  // (!cir.ptr<!ty_22aie3A3Atile3C12C_43E22>, !cir.ptr<!ty_22anon2E122>) -> ()
  // loc(#loc63)
  void program(auto&& code) __attribute__((annotate("aie.tile.program"))) {
    // Use this since std::function crashes ClangIR 2024/09/12
    // tile_code<decltype(&code)> = &code;
    // Just to instantiate the lambda body while waiting for std::function
    code();
  }

  // Create a channel to communicate between 2 tiles.
  template <typename ValueType, std::size_t Size, typename ToTile>
  channel<ValueType, Size, tile, ToTile> channel_to(ToTile& to_tile,
                                                    std::size_t capacity)
      __attribute__((annotate("aie.device.channel"))) {
    return { *this, to_tile, capacity };
  }
};

// template <int X, int Y> inline constexpr tile<X, Y> tile;

template <typename Storage> struct tile_handle {
  Storage tile_memory;
  constexpr tile_handle(Storage tile_memory)
      : tile_memory { tile_memory } {};
  constexpr auto& mem() { return tile_memory; }
};

/*
template <typename ValueType, std::size_t Size, typename FromTile,
          typename ToTile>
channel<ValueType, Size>(FromTile& from_tile, ToTile& to_tile,
                         std::size_t capacity)
    ->channel<ValueType, Size, FromTile, ToTile>;
*/
// Inject in aie:: to ease use
enum : std::int8_t {
  npu1 = 42, //< AIE2 device found in RyzenAI 9 7940HS
  npu2,
  npu3,
  npu4,
  pacifica //< Fancy AIE device found on Californian shore
};

// Abstraction representing an AIE device in the system.
//
// DeviceModel specifies an architecture model, such as aie::npu1.
//
// Typically compiled as:
// !ty_aie3A3Adevice3Caie3A3Anpu12C_aie3A3A28lambda_at_2E2Faie2B2B2Ehpp3A763A54293E
// = !cir.struct<struct "aie::device<aie::npu1, aie::(lambda at
// ./aie++.hpp:76:54)>" {!cir.int<u, 8>}>
//
// "Unique" with the lambda is used to generate a different type for multiple
// instantiation, so we can have a design with several accelerators of the same
// type
template <auto DeviceModel = npu1, typename Unique = decltype([] {})>
struct device {
  // Get the tile of a device. X is the column of the tile, Y is the row of the
  // tile.
  template <int X, int Y>
  tile<X, Y, device> tile()
      __attribute__((annotate("aie.device.tile", X, Y, DeviceModel,
                              std::to_underlying(DeviceModel)))) {
    return {};
  }

  void constexpr run() {}
};

template <typename T> struct tile_storage_t {
  using content = T;
};

template <typename T> inline constexpr tile_storage_t<T> tile_storage;

} // namespace aie
