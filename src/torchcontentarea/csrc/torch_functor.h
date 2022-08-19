#include <torch/torch.h>
#include <stdexcept>

enum Device
{
    none = 0,
    cpu = 1,
    cuda = 2,
};

inline Device operator|(Device a, Device b)
{
    return static_cast<Device>(static_cast<int>(a)|static_cast<int>(b));
}

template<typename FunctorType, typename ReturnType, typename ...ArgTypes>
class TorchFunctor
{
public:

    static ReturnType invoke(ArgTypes... args)
    {
        Device device = start_unpack(args...);

        switch (device)
        {
            case (Device::cpu): return FunctorType::cpu_implementation(args...);
            case (Device::cuda): return FunctorType::cuda_implementation(args...);
            default: throw std::invalid_argument("Torch tensors found on different devices!");
        }
    }

private:

    template<typename ...Types>
    static inline Device start_unpack(Types&... args)
    {
        return unpack(Device::none, args...);
    }

    template<typename Type, typename ...Types>
    static inline Device unpack(Device device, Type& arg, Types&... args)
    {
        device = device | check_device(arg);
        return unpack(device, args...);
    }

    template<typename Type>
    static inline Device unpack(Device device, Type& arg)
    {
        return device | check_device(arg);
    }

    template<typename Type>
    static inline Device check_device(Type& arg)
    {
        return Device::none;
    }

    static inline Device check_device(torch::Tensor& arg)
    {
        arg = arg.contiguous();
        return arg.device().is_cuda() ? Device::cuda : Device::cpu;
    }
};
