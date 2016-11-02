using System.Threading.Tasks;

namespace TextToMotionWeb.Services
{
    public interface ISmsSender
    {
        Task SendSmsAsync(string number, string message);
    }
}
