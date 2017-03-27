using Microsoft.AspNetCore.Mvc;

namespace TextToMotionWeb.Controllers
{
    public class TextToMotionController : Controller
    {
        public IActionResult Index()
        {
            return View();
        }
    }
}