using Microsoft.AspNetCore.Mvc;

namespace TextToMotionWeb.Controllers
{
    public class DemoController : Controller
    {
        public IActionResult ScreencapDemo()
        {
            return View();
        }

        public IActionResult Index()
        {
            return RedirectToAction("ScreencapDemo");
        }
    }
}